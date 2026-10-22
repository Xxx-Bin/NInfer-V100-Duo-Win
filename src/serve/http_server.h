#pragma once

#include "serve/generation_service.h"
#include "serve/response_store.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"
#include "serve/system_monitor.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace ninfer::serve {

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

class HttpServer {
public:
    explicit HttpServer(ServeOptions options);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;
    void handle_metrics(const httplib::Request& req, httplib::Response& res);
    void handle_web_asset(const httplib::Request& req, httplib::Response& res) const;

    // The process-wide console logger serializes lines from request and reporter threads.
    void log_line(const std::string& line);
    void log_request_start(const RequestLogContext& context);
    void log_request_rejected(const RequestRejectionLogContext& context);
    void log_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void log_request_error(const RequestLogContext& context, const std::string& message);
    void log_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();
    // The fixed-rate owner of every /metrics-derived number. Keeping sampling off the HTTP
    // threads makes the rate window and history cadence independent of how many browsers poll
    // and how fast each one does.
    void run_monitor_sampler();
    void stop_monitor_sampler();

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    std::string public_model_id_;
    ResponseStore response_store_;
    JsonlRequestLog request_jsonl_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;

    // Web panel: engine snapshots captured at attach() plus the fixed-rate monitor sampler.
    ninfer::LoadSummary load_summary_{};
    ninfer::MemorySummary memory_summary_{};
    bool has_load_summary_ = false;
    SystemMonitor system_monitor_;
    std::mutex monitor_mutex_;
    std::condition_variable monitor_cv_;
    std::thread monitor_thread_;
    bool monitor_stopping_ = false;
    // Last sample published by run_monitor_sampler(). /metrics only copies this state, so a
    // request never drives the rate window.
    ninfer::RuntimeStats monitor_stats_{};
    HostReading monitor_system_{};
    double monitor_tok_s_                            = 0.0;
    double monitor_prefill_tok_s_                    = 0.0;
    double monitor_cache_hit_tok_s_                  = 0.0;
    std::uint64_t monitor_prefill_interval_tokens_   = 0;
    std::uint64_t monitor_decode_interval_tokens_    = 0;
    std::uint64_t monitor_cache_hit_interval_tokens_ = 0;
    // Last completed request's measured rates, the panel's only exact per-request figures: their
    // denominators are the phases' own wall time, not a sampling interval.
    RequestRates last_rates_{};
    bool has_last_rates_ = false;
    // Session maxima of the per-request rates above. -1.0 marks "no request has measured one yet".
    // These are the panel's peak figures: a sampled counter can only report multiples of the
    // prefill chunk it lands in, so no sampled series can stand in for a per-request measurement.
    double peak_prefill_tok_s_ = -1.0;
    double peak_decode_tok_s_  = -1.0;
    // TTFT (request preparation to first token) of the last completed request, for the panel.
    std::optional<double> last_ttft_ = std::nullopt;
    // Fixed-size rings of the last kMonitorHistorySamples samples, served back as the panel's
    // series. Every engine ring holds raw per-interval token counts plus the interval length;
    // a rate is only ever formed by dividing one by the other, so no ring stores a stall the
    // sampler smoothed over. cache_hit_tokens is the reused-prompt count of the interval.
    static constexpr std::size_t kMonitorHistorySamples = 120;
    // Intervals averaged into the live rate. One interval can only report 0 or a whole prefill
    // chunk divided by ~1s, which reads as roughly 2x the real rate at this chunk size; summing
    // whole intervals is what removes that quantization.
    static constexpr std::size_t kMonitorRateWindowSamples = 3;
    struct MonitorRing {
        std::deque<double> values;
        void push(double value) {
            values.push_back(value);
            if (values.size() > kMonitorHistorySamples) { values.pop_front(); }
        }
    };
    MonitorRing ring_prefill_tokens_;
    MonitorRing ring_decode_tokens_;
    MonitorRing ring_interval_seconds_;
    MonitorRing ring_cache_hit_tokens_;
    MonitorRing ring_cpu_util_;
    MonitorRing ring_ram_used_;
};

} // namespace ninfer::serve
