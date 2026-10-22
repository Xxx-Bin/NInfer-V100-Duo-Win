#include "serve/http_server.h"

#include "product/speculative_options.h"
#include "serve/anthropic_schema.h"
#include "serve/console_log.h"
#include "serve/openai_schema.h"
#include "serve/request_log.h"
#include "serve/translate.h"
#include "serve/web_assets.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

struct StreamingRequest {
    explicit StreamingRequest(PreparedRequest request) : prepared(std::move(request)) {}

    PreparedRequest prepared;
    std::atomic<bool> cancelled{false};
    bool started = false;
};

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

void write_stream_item(httplib::DataSink& sink, StreamingRequest& request,
                       const std::string& item) {
    if (request.cancelled.load(std::memory_order_acquire) ||
        (sink.is_writable && !sink.is_writable()) || !sink.write(item.data(), item.size())) {
        request.cancelled.store(true, std::memory_order_release);
        throw ClientDisconnected();
    }
}

void set_owned_content(httplib::Response& response, std::string body,
                       std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.hold_resource(std::move(lifetime));
}

void write_error(httplib::Response& res, const ApiError& error) {
    res.status = error.status;
    res.set_content(make_error_body(error), "application/json");
}

// Anthropic-shaped error body ({"type":"error","error":{...}}), used by the
// /v1/messages endpoints so Claude clients see the error format they expect.
void write_messages_error(httplib::Response& res, const ApiError& error) {
    res.status = error.status;
    res.set_content(make_messages_error_body(error), "application/json");
}

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_error(res, error);
}

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .scheduler         = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.scheduler.running_requests != 0 ||
           report.scheduler.waiting_requests != 0;
}

std::string_view unstreamed_content(const GenerationOutcome& outcome) {
    if (outcome.streamed_content_bytes > outcome.text.size()) {
        throw std::logic_error("streamed content exceeds terminal content");
    }
    return std::string_view(outcome.text).substr(outcome.streamed_content_bytes);
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

std::string_view web_content_type(std::string_view path) {
    if (ends_with(path, ".html")) { return "text/html; charset=utf-8"; }
    if (ends_with(path, ".css")) { return "text/css; charset=utf-8"; }
    if (ends_with(path, ".js")) { return "text/javascript; charset=utf-8"; }
    if (ends_with(path, ".svg")) { return "image/svg+xml"; }
    if (ends_with(path, ".woff2")) { return "font/woff2"; }
    if (ends_with(path, ".txt")) { return "text/plain; charset=utf-8"; }
    return "application/octet-stream";
}

// Seconds -> optional JSON number, dropping non-finite values instead of emitting invalid JSON.
nlohmann::json rate_json(double value) {
    return value >= 0.0 ? nlohmann::json(value) : nlohmann::json(nullptr);
}

nlohmann::json ring_json(const std::deque<double>& values) {
    nlohmann::json out = nlohmann::json::array();
    for (const double value : values) { out.push_back(rate_json(value)); }
    return out;
}

// Mean throughput over the last `window` published intervals. A phase's token counter moves once
// per prefill chunk or decode round, so one interval can only report 0 or a whole chunk divided by
// ~1s - at a 2048-token chunk that reads as roughly twice the real rate. Summing whole intervals is
// what removes that quantization. A stopped phase returns 0 so the panel falls back to its exact
// per-request figure instead of showing a decaying ramp.
double window_rate(const std::deque<double>& tokens, const std::deque<double>& seconds,
                   std::size_t window, bool active) {
    if (!active) { return 0.0; }
    const std::size_t available = std::min(tokens.size(), seconds.size());
    const std::size_t count     = std::min(available, window);
    double total_tokens         = 0.0;
    double total_seconds        = 0.0;
    for (std::size_t i = available - count; i < available; ++i) {
        if (!(seconds[i] > 0.0)) { continue; }
        total_tokens += std::max(0.0, tokens[i]);
        total_seconds += seconds[i];
    }
    return total_seconds > 0.0 ? total_tokens / total_seconds : 0.0;
}

// Basename of the artifact path, so the panel can name the exact checkpoint file the engine
// loaded without exposing a full local path. Empty for a missing path.
std::string artifact_base_name(const std::string& path) {
    std::filesystem::path p(path);
    const std::string name = p.filename().string();
    return name.empty() ? path : name;
}

} // namespace

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    if (response.status != 413 || !response.body.empty()) {
        return httplib::Server::HandlerResponse::Unhandled;
    }

    ApiError error;
    error.status  = 413;
    error.type    = "invalid_request_error";
    error.code    = "request_too_large";
    error.message = "request body exceeds the configured payload limit of " +
                    std::to_string(options.max_request_bytes) + " bytes";
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_messages_error(response, error);
    } else {
        write_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

HttpServer::HttpServer(ServeOptions options)
    : options_(std::move(options)),
      response_store_(options_.response_store_max_records, options_.response_store_max_bytes),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path) {
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, queued_requests);
    };
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

void HttpServer::log_line(const std::string& line) {
    write_console_log(ConsoleLogLevel::Info, line);
}

void HttpServer::log_request_start(const RequestLogContext& context) {
    log_line(format_request_start(context));
    request_jsonl_.write_request_start(context);
}

void HttpServer::log_request_rejected(const RequestRejectionLogContext& context) {
    log_line(format_request_rejected(context));
    request_jsonl_.write_request_rejected(context);
}

void HttpServer::log_request_done(const RequestLogContext& context,
                                  const GenerationOutcome& outcome) {
    log_line(format_request_done(context, outcome));
    request_jsonl_.write_request_done(context, outcome);
    {
        std::lock_guard lock(monitor_mutex_);
        last_rates_     = request_rates(outcome);
        has_last_rates_ = true;
        if (last_rates_.prefill_tok_s > peak_prefill_tok_s_) {
            peak_prefill_tok_s_ = last_rates_.prefill_tok_s;
        }
        if (last_rates_.decode_tok_s > peak_decode_tok_s_) {
            peak_decode_tok_s_ = last_rates_.decode_tok_s;
        }
        // TTFT (request preparation to first token) of the last completed request, for the panel.
        if (outcome.metrics.ttft_seconds > 0.0) { last_ttft_ = outcome.metrics.ttft_seconds; }
    }
}

void HttpServer::log_request_error(const RequestLogContext& context, const std::string& message) {
    log_line(format_request_error(context, message));
    request_jsonl_.write_request_error(context, message);
}

void HttpServer::log_throughput(const ThroughputReport& report) {
    log_line(format_throughput(report));
    request_jsonl_.write_throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_for(lock, interval, [this] { return stats_stopping_; })) { break; }
        }

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { log_throughput(report); }
        previous      = current;
        previous_time = now;
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    if (tail.computed_prefill_tokens != 0 || tail.committed_decode_tokens != 0 ||
        tail.decode_rounds != 0) {
        log_throughput(tail);
    }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::run_monitor_sampler() {
    using Clock = std::chrono::steady_clock;
    constexpr auto kSampleInterval = std::chrono::seconds(1);

    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();

    // Publish one immediate host reading so the panel is not blank for the first interval.
    {
        HostReading system = system_monitor_.sample();
        std::lock_guard lock(monitor_mutex_);
        monitor_stats_  = previous;
        monitor_system_ = std::move(system);
        monitor_tok_s_                     = 0.0;
        monitor_prefill_tok_s_             = 0.0;
        monitor_cache_hit_tok_s_           = 0.0;
        monitor_prefill_interval_tokens_   = 0;
        monitor_decode_interval_tokens_    = 0;
        monitor_cache_hit_interval_tokens_ = 0;
        ring_prefill_tokens_.push(0.0);
        ring_decode_tokens_.push(0.0);
        ring_interval_seconds_.push(0.0);
        ring_cache_hit_tokens_.push(0.0);
        ring_cpu_util_.push(monitor_system_.cpu_util_available ? monitor_system_.cpu_util
                                                               : -1.0);
        ring_ram_used_.push(monitor_system_.ram_total != 0
                                ? static_cast<double>(monitor_system_.ram_used)
                                : -1.0);
    }

    for (;;) {
        {
            std::unique_lock lock(monitor_mutex_);
            if (monitor_cv_.wait_for(lock, kSampleInterval, [this] { return monitor_stopping_; })) {
                break;
            }
        }

        const ninfer::RuntimeStats stats = service_->runtime_stats();
        const Clock::time_point now      = Clock::now();
        const double seconds = std::chrono::duration<double>(now - previous_time).count();
        std::uint64_t prefill_interval_tokens   = 0;
        std::uint64_t decode_interval_tokens    = 0;
        std::uint64_t cache_hit_interval_tokens = 0;
        if (seconds > 0.0) {
            prefill_interval_tokens =
                stats.computed_prefill_tokens - previous.computed_prefill_tokens;
            decode_interval_tokens =
                stats.committed_decode_tokens - previous.committed_decode_tokens;
            cache_hit_interval_tokens =
                stats.reused_prompt_tokens - previous.reused_prompt_tokens;
        }

        HostReading system = system_monitor_.sample();
        std::lock_guard lock(monitor_mutex_);
        monitor_stats_  = stats;
        monitor_system_ = std::move(system);
        monitor_prefill_interval_tokens_   = prefill_interval_tokens;
        monitor_decode_interval_tokens_    = decode_interval_tokens;
        monitor_cache_hit_interval_tokens_ = cache_hit_interval_tokens;
        // Rings carry raw counts and interval lengths; every rate the panel shows is formed by
        // dividing one by the other, never by carrying a previous interval forward.
        ring_prefill_tokens_.push(static_cast<double>(prefill_interval_tokens));
        ring_decode_tokens_.push(static_cast<double>(decode_interval_tokens));
        ring_interval_seconds_.push(seconds > 0.0 ? seconds : 0.0);
        ring_cache_hit_tokens_.push(static_cast<double>(cache_hit_interval_tokens));
        ring_cpu_util_.push(monitor_system_.cpu_util_available ? monitor_system_.cpu_util
                                                              : -1.0);
        ring_ram_used_.push(monitor_system_.ram_total != 0
                                ? static_cast<double>(monitor_system_.ram_used)
                                : -1.0);
        monitor_prefill_tok_s_ =
            window_rate(ring_prefill_tokens_.values, ring_interval_seconds_.values,
                        kMonitorRateWindowSamples, stats.prefilling_requests != 0);
        monitor_tok_s_ = window_rate(ring_decode_tokens_.values, ring_interval_seconds_.values,
                                     kMonitorRateWindowSamples, stats.decode_ready_requests != 0);
        // Prefix reuse lands in one burst per completed request and has no per-request fallback, so
        // it decays with the window instead of dropping at the phase boundary.
        monitor_cache_hit_tok_s_ =
            window_rate(ring_cache_hit_tokens_.values, ring_interval_seconds_.values,
                        kMonitorRateWindowSamples, true);
        previous      = stats;
        previous_time = now;
    }
}

void HttpServer::stop_monitor_sampler() {
    if (!monitor_thread_.joinable()) { return; }
    {
        std::lock_guard lock(monitor_mutex_);
        monitor_stopping_ = true;
    }
    monitor_cv_.notify_one();
    monitor_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        // The panel shell itself is public; every data route (including /metrics) still needs
        // the key, and the page asks for it under About > Settings.
        const bool public_shell =
            req.path == "/" || req.path.rfind("/web/", 0) == 0 || req.path == "/favicon.ico";
        if (options_.api_key.empty() || req.method == "OPTIONS" || req.path == "/health" ||
            public_shell) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            req.get_header_value("Authorization") == ("Bearer " + options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_messages_error(res, error);
            } else {
                write_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                write_error(res, e.error());
            } catch (const std::exception& e) { write_exception(res, e); } catch (...) {
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                write_error(res, error);
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        nlohmann::json body{{"status", "ok"}};
        if (!public_model_id_.empty()) { body["model"] = public_model_id_; }
        res.set_content(body.dump(), "application/json");
    });
    server_.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        handle_metrics(req, res);
    });
    // The embedded web panel: "/" serves the shell, "/web/<path>" its assets.
    server_.Get("/", [this](const httplib::Request& req, httplib::Response& res) {
        handle_web_asset(req, res);
    });
    server_.Get(R"(/web/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_web_asset(req, res);
    });
    // A bare browser request for the site icon: the panel's own SVG stands in for it.
    server_.Get("/favicon.ico", [this](const httplib::Request& req, httplib::Response& res) {
        handle_web_asset(req, res);
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now()), "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now()), "application/json");
}

void HttpServer::handle_web_asset(const httplib::Request& req, httplib::Response& res) const {
    std::string path;
    if (req.path == "/") {
        path = "index.html";
    } else if (req.path == "/favicon.ico") {
        path = "favicon.svg";
    } else if (req.path.rfind("/web/", 0) == 0) {
        path = req.path.substr(5);
    } else {
        res.status = 404;
        return;
    }
    // The asset set is a fixed embedded list; reject anything outside it (no traversal possible,
    // since nothing reads the filesystem here).
    for (const web::WebAsset& asset : web::assets()) {
        if (asset.path == path) {
            res.set_content(std::string(asset.content), std::string(web_content_type(path)));
            return;
        }
    }
    res.status = 404;
}

void HttpServer::handle_metrics(const httplib::Request&, httplib::Response& res) {
    ninfer::RuntimeStats stats;
    HostReading system;
    RequestRates last_rates;
    bool has_last_rates                      = false;
    double tok_s                             = 0.0;
    double prefill_tok_s                     = 0.0;
    double cache_hit_tok_s                   = 0.0;
    std::uint64_t prefill_interval_tokens    = 0;
    std::uint64_t decode_interval_tokens     = 0;
    std::uint64_t cache_hit_interval_tokens  = 0;
    std::optional<double> last_ttft_opt = std::nullopt;
    double peak_prefill_tok_s                = -1.0;
    double peak_decode_tok_s                 = -1.0;
    std::deque<double> h_prefill_tokens, h_decode_tokens, h_seconds, h_cache_hit_tokens, h_cpu,
        h_ram;
    // The sampler owns the rate window and the sparkline cadence; this handler only copies one
    // consistent snapshot. A browser poll therefore cannot shorten the interval.
    {
        std::lock_guard lock(monitor_mutex_);
        stats                     = monitor_stats_;
        system                    = monitor_system_;
        tok_s                     = monitor_tok_s_;
        prefill_tok_s             = monitor_prefill_tok_s_;
        cache_hit_tok_s           = monitor_cache_hit_tok_s_;
        prefill_interval_tokens   = monitor_prefill_interval_tokens_;
        decode_interval_tokens    = monitor_decode_interval_tokens_;
        cache_hit_interval_tokens = monitor_cache_hit_interval_tokens_;
        last_rates                = last_rates_;
        has_last_rates            = has_last_rates_;
        last_ttft_opt             = last_ttft_;
        peak_prefill_tok_s        = peak_prefill_tok_s_;
        peak_decode_tok_s         = peak_decode_tok_s_;
        h_prefill_tokens   = ring_prefill_tokens_.values;
        h_decode_tokens    = ring_decode_tokens_.values;
        h_seconds          = ring_interval_seconds_.values;
        h_cache_hit_tokens = ring_cache_hit_tokens_.values;
        h_cpu              = ring_cpu_util_.values;
        h_ram              = ring_ram_used_.values;
    }

    // Live scheduler state: the prefill lane drives "Reading", decodable slots drive
    // "Generating"; queued requests are reported separately. Prefill and decode run
    // concurrently, so both rates are always reported - they read 0 while nothing moves. Each is
    // the mean over the sampler's last few intervals, so a phase that just stopped reads 0 rather
    // than trailing its own decay.
    const bool generating = stats.decode_ready_requests != 0;
    const bool reading    = stats.prefilling_requests != 0;
    const char* state     = reading ? "reading" : generating ? "generating" : "idle";

    // The engine admits a request against `max_context`; `effective_max_context` is only the
    // ceiling that value was validated against (the variant's native capacity, or the YaRN
    // window). Both are reported, because they answer different questions.
    const std::uint32_t admitted_max_context = memory_summary_.max_context != 0
                                                   ? memory_summary_.max_context
                                                   : options_.max_context;
    // KV pool size: the page-aligned Main KV capacity resolved at load. `kv_payload_bytes` is the
    // total bytes that pool occupies on rank 0; the page-group figures bracket how much of the
    // pool the model can actually fill before the next page allocation.
    const bool spec_enabled = options_.speculative.backend != SpeculativeBackend::None;
    const std::string artifact_name = artifact_base_name(options_.artifact_path);
    nlohmann::json engine = nlohmann::json{
        {"version", kNinferVersion},
        {"model", public_model_id_},
        {"artifact", artifact_name},
        {"tp", load_summary_.tp},
        {"devices", options_.devices},
        {"max_context", admitted_max_context},
        {"max_context_ceiling", load_summary_.effective_max_context},
        {"max_concurrency", options_.max_concurrency},
        {"max_pending_requests", options_.max_pending_requests},
        {"prefill_chunk", options_.prefill_chunk},
        {"kv_pool_tokens", memory_summary_.kv_capacity},
        {"kv_pool_bytes", memory_summary_.kv_payload_bytes},
        {"kv_page_groups", memory_summary_.kv_capacity_page_groups},
        {"kv_page_groups_max", memory_summary_.kv_capacity_max_page_groups},
        {"kv_cache", options_.kv_cache == KvCacheStorage::Int8Group64 ? "int8_g64" : "bfloat16"},
        {"vision", options_.enable_vision},
        {"vision_max_tokens", std::min(options_.max_context, options_.vision_max_tokens)},
        {"cuda_graph", options_.use_cuda_graph},
        {"prefix_reuse", options_.allow_prefix_reuse},
        // Speculative decoding: the draft backend type and the configured draft window. One draft
        // model per server, so "type and quantity" is the backend name plus its window.
        {"speculative_backend", product::speculative_backend_name(options_.speculative.backend)},
        {"speculative_draft_tokens", spec_enabled ? options_.speculative.draft_tokens : 0},
        {"speculative_draft_count", spec_enabled ? 1 : 0},
    };
    if (!load_summary_.weights_id.empty()) { engine["weights_id"] = load_summary_.weights_id; }

    nlohmann::json hardware{
        {"cpu_util", system.cpu_util_available ? nlohmann::json(system.cpu_util)
                                               : nlohmann::json(nullptr)},
        {"ram_total", system.ram_total},
        {"ram_used", system.ram_total != 0 ? nlohmann::json(system.ram_used)
                                           : nlohmann::json(nullptr)},
    };

    nlohmann::json hardware_static{
        {"cpu_name", system_monitor_.host().cpu_name},
        {"threads", system_monitor_.host().threads},
    };

    nlohmann::json history{
        {"prefill_tokens", ring_json(h_prefill_tokens)},
        {"decode_tokens", ring_json(h_decode_tokens)},
        {"interval_seconds", ring_json(h_seconds)},
        {"cache_hit_tokens", ring_json(h_cache_hit_tokens)},
        {"cpu_util", ring_json(h_cpu)},
        {"ram_used", ring_json(h_ram)},
    };

    // Session figures the panel's engine cards show. Peaks are the per-request measurements the
    // engine itself timed, not the sampled counters: one sampling interval can only report a whole
    // prefill chunk divided by its length, which is several times the real rate above one chunk per
    // interval. The panel forms every shown average from the raw per-interval token counts against
    // their interval lengths. The cache-hit rate is the session share of prompt tokens answered
    // from the prefix cache rather than recomputed; null while no prompt has completed, so the panel
    // can show a dash instead of a 0% that would read as "the cache is useless". TTFT is the last
    // completed request's wall time from preparation to its first token.
    const std::uint64_t computed_prefill = stats.computed_prefill_tokens;
    const std::uint64_t reused_prompt    = stats.reused_prompt_tokens;
    const std::uint64_t prompt_total     = computed_prefill + reused_prompt;
    nlohmann::json cache_hit_rate = nlohmann::json(nullptr);
    if (prompt_total != 0) {
        cache_hit_rate = static_cast<double>(reused_prompt) / static_cast<double>(prompt_total);
    }
    nlohmann::json ttft_seconds = nlohmann::json(nullptr);
    if (last_ttft_opt.has_value()) { ttft_seconds = *last_ttft_opt; }
    nlohmann::json performance = nlohmann::json{
        {"cache_hit_rate", std::move(cache_hit_rate)},
        {"prompt_tokens_computed", computed_prefill},
        {"prompt_tokens_reused", reused_prompt},
        {"ttft_seconds", std::move(ttft_seconds)},
    };

    nlohmann::json body{
        {"live",
         {{"state", state},
          {"queued", stats.waiting_requests},
          {"running", stats.running_requests},
          {"tok_s", tok_s},
          {"prefill_tok_s", prefill_tok_s},
          {"prefill_interval_tokens", prefill_interval_tokens},
          {"decode_interval_tokens", decode_interval_tokens},
          {"cache_hit_tokens", stats.reused_prompt_tokens},
          {"cache_hit_tok_s", cache_hit_tok_s},
          {"cache_hit_interval_tokens", cache_hit_interval_tokens}}},
        {"last",
         {{"available", has_last_rates},
          {"prefill_tok_s", rate_json(has_last_rates ? last_rates.prefill_tok_s : -1.0)},
          {"tok_s", rate_json(has_last_rates ? last_rates.decode_tok_s : -1.0)},
          {"peak_prefill_tok_s", rate_json(peak_prefill_tok_s)},
          {"peak_tok_s", rate_json(peak_decode_tok_s)},
          {"cache_hit_tokens",
           has_last_rates ? nlohmann::json(last_rates.cache_hit_tokens) : nlohmann::json(nullptr)}}},
        {"hardware", std::move(hardware)},
        {"hardware_static", std::move(hardware_static)},
        {"engine", std::move(engine)},
        {"performance", std::move(performance)},
        {"history", std::move(history)},
    };
    res.set_content(body.dump(), "application/json");
}

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_error(res, error);
        return;
    }

    GenerationRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_chat_completion_request(body, limits);
        if (request.model != public_model_id_) {
            ApiError error;
            error.status  = 404;
            error.type    = "invalid_request_error";
            error.code    = "model_not_found";
            error.message = "model '" + request.model + "' not found";
            throw ApiException(std::move(error));
        }
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
    } catch (const ApiException& e) {
        log_request_rejected(make_request_rejection_log_context(req_id, "openai_chat_completions",
                                                                request, e.error()));
        write_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        log_request_rejected(
            make_request_rejection_log_context(req_id, "openai_chat_completions", request, error));
        write_error(res, error);
        return;
    }

    const std::string id       = new_chat_completion_id();
    const std::int64_t created = unix_time_now();
    const std::string model    = request.model;

    const RequestLogContext log_context =
        make_request_log_context(req_id, "openai_chat_completions", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr, [&req] {
                return req.is_connection_alive && !req.is_connection_alive();
            });
            log_request_done(log_context, outcome);
            CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            usage.cached_tokens = std::clamp(
                static_cast<int>(outcome.metrics.prefix_cache_hit_tokens), 0, outcome.prompt_tokens);
            std::string response_body;
            if (!outcome.tool_calls.empty()) {
                response_body = make_chat_completion_tool_response(
                    id, model, created, outcome.text, outcome.reasoning, outcome.tool_calls, usage);
            } else {
                response_body = make_chat_completion_response(
                    id, model, created, outcome.text, outcome.reasoning,
                    finish_reason_wire(outcome.finish_reason), usage);
            }
            set_owned_content(res, std::move(response_body), prepared.lifetime);
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            throw;
        }
        return;
    }

    auto stream              = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool include_usage = stream->prepared.include_usage;
    const bool tool_capable  = stream->prepared.tool_capable;

    // SSE hints: disable client/proxy caching and reverse-proxy response buffering
    // so tokens flush immediately. Content-Type is set by the chunked provider.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, created, model, include_usage, tool_capable,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            try {
                write_stream_item(sink, *stream,
                                  make_chat_chunk_role(id, model, created, include_usage));
                StreamSink output;
                output.on_content = [&](const std::string& text) {
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_content(id, model, created, text, include_usage));
                };
                output.on_reasoning = [&](const std::string& text) {
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_reasoning(id, model, created, text, include_usage));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                const std::string_view remaining = unstreamed_content(outcome);
                if (!outcome.tool_calls.empty()) {
                    if (!remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_tool_calls(
                                          id, model, created, outcome.tool_calls, include_usage));
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_final(id, model, created, "tool_calls", include_usage));
                } else {
                    if (tool_capable && !remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_final(id, model, created,
                                              finish_reason_wire(outcome.finish_reason),
                                              include_usage));
                }
                if (include_usage) {
                    CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
                    usage.cached_tokens = std::clamp(
                        static_cast<int>(outcome.metrics.prefix_cache_hit_tokens),
                        0, outcome.prompt_tokens);
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_usage(id, model, created, usage));
                }
                write_stream_item(sink, *stream, sse_done());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = e.what();
                try {
                    write_stream_item(sink, *stream, sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }
    try {
        RequestLimits limits;
        limits.default_max_tokens       = options_.default_max_tokens;
        const GenerationRequest request = parse_messages_request(body, limits);
        const int input_tokens          = service_->count_prompt_tokens(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
        res.set_content(make_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        write_messages_error(res, error);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }

    GenerationRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        // The Anthropic endpoint accepts any `model` string (Claude Code sends real
        // Claude model names) and echoes it back; it never 404s on model id.
        request = parse_messages_request(body, limits);
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        write_messages_error(res, error);
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
    } catch (const ApiException& e) {
        log_request_rejected(
            make_request_rejection_log_context(req_id, "anthropic_messages", request, e.error()));
        write_messages_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        log_request_rejected(
            make_request_rejection_log_context(req_id, "anthropic_messages", request, error));
        write_messages_error(res, error);
        return;
    }

    const std::string id    = new_message_id();
    const std::string model = request.model; // echo the requested model
    const int input_tokens  = prepared.prompt_tokens;

    const RequestLogContext log_context =
        make_request_log_context(req_id, "anthropic_messages", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr, [&req] {
                return req.is_connection_alive && !req.is_connection_alive();
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            const char* stop_reason =
                messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
            set_owned_content(res,
                              make_messages_response(id, model, outcome.text, outcome.reasoning,
                                                     outcome.tool_calls, stop_reason, usage),
                              prepared.lifetime);
        } catch (const ApiException& e) {
            log_request_error(log_context, e.error().message);
            write_messages_error(res, e.error());
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = e.what();
            write_messages_error(res, error);
        }
        return;
    }

    auto stream             = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool tool_capable = stream->prepared.tool_capable;

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, model, input_tokens, tool_capable,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;

            int next_index     = 0;
            bool thinking_open = false;
            int thinking_index = -1;
            bool text_open     = false;
            int text_index     = -1;
            try {
                write_stream_item(sink, *stream, make_message_start(id, model, input_tokens));

                StreamSink output;
                output.on_reasoning = [&](const std::string& text) {
                    if (!thinking_open) {
                        thinking_index = next_index++;
                        thinking_open  = true;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_thinking(thinking_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_thinking(thinking_index, text));
                };
                output.on_content = [&](const std::string& text) {
                    if (thinking_open) {
                        write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                        thinking_open = false;
                    }
                    if (!text_open) {
                        text_index = next_index++;
                        text_open  = true;
                        write_stream_item(sink, *stream, make_content_block_start_text(text_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_text(text_index, text));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                const std::string_view remaining = unstreamed_content(outcome);

                if (thinking_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                    thinking_open = false;
                }
                if (text_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(text_index));
                    text_open = false;
                }

                if (tool_capable) {
                    if (!remaining.empty()) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream, make_content_block_start_text(idx));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_text(idx, std::string(remaining)));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                    for (const ToolCall& call : outcome.tool_calls) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_tool_use(idx, call));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_tool_json(idx, call.arguments_json));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                }

                if (next_index == 0) {
                    const int idx = next_index++;
                    write_stream_item(sink, *stream, make_content_block_start_text(idx));
                    write_stream_item(sink, *stream, make_content_block_stop(idx));
                }

                const char* stop_reason =
                    messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
                write_stream_item(sink, *stream,
                                  make_message_delta(stop_reason, outcome.completion_tokens));
                write_stream_item(sink, *stream, make_message_stop());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, messages_sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = e.what();
                try {
                    write_stream_item(sink, *stream, messages_sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_id);
    load_summary_                  = load;
    memory_summary_                = service.memory_summary();
    has_load_summary_              = true;
    service_                       = &service;
    request_jsonl_.write_server_start(options_, service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    monitor_stopping_ = false;
    monitor_thread_   = std::thread([this] { run_monitor_sampler(); });
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        stop_monitor_sampler();
        return result;
    } catch (...) {
        stop_stats_reporter();
        stop_monitor_sampler();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

} // namespace ninfer::serve
