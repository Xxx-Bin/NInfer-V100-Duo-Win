#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {
namespace {

std::string take_string(artifact::MaterializedArtifact& materialized,
                        artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

// Text-only artifacts omit the Vision preprocessor JSON. Binding therefore has to ask the
// artifact whether the object exists instead of assuming the full contract.
std::optional<artifact::ObjectHandle> bind_optional_raw_resource(artifact::Binder& binder,
                                                                std::string_view name) {
    if (binder.find(name) == nullptr) { return std::nullopt; }
    return artifact::bind_raw_resource(binder, name);
}

} // namespace

FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder) {
    const auto preprocessor_config =
        bind_optional_raw_resource(binder, "frontend/preprocessor_config.json");
    const auto video_preprocessor_config =
        bind_optional_raw_resource(binder, "frontend/video_preprocessor_config.json");
    return FrontendResourcePlan{
        .tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json"),
        .tokenizer_config_json =
            artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json"),
        .chat_template_jinja = artifact::bind_raw_resource(binder, "frontend/chat_template.jinja"),
        .generation_config_json =
            artifact::bind_raw_resource(binder, "frontend/generation_config.json"),
        .preprocessor_config_json = preprocessor_config.value_or(artifact::ObjectHandle{}),
        .video_preprocessor_config_json =
            video_preprocessor_config.value_or(artifact::ObjectHandle{}),
        .has_preprocessor_config       = preprocessor_config.has_value(),
        .has_video_preprocessor_config = video_preprocessor_config.has_value(),
    };
}

FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                          const FrontendResourcePlan& plan) {
    return FrontendResources{
        .tokenizer_json           = take_string(materialized, plan.tokenizer_json),
        .tokenizer_config_json    = take_string(materialized, plan.tokenizer_config_json),
        .chat_template_jinja      = take_string(materialized, plan.chat_template_jinja),
        .generation_config_json   = take_string(materialized, plan.generation_config_json),
        .preprocessor_config_json = plan.has_preprocessor_config
                                        ? take_string(materialized, plan.preprocessor_config_json)
                                        : std::string{},
        .video_preprocessor_config_json =
            plan.has_video_preprocessor_config
                ? take_string(materialized, plan.video_preprocessor_config_json)
                : std::string{},
    };
}

} // namespace ninfer::targets::qwen3_6
