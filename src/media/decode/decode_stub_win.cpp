// Stub of the FFmpeg-backed image/video decoder for platform bring-up builds.
//
// Selected by CMake's NINFER_STUB_MEDIA=ON. It exists so every product target and its link graph
// can be built on a platform without an FFmpeg toolchain (the Windows port's Phase 2). The symbols
// match src/media/decode/decode.cpp exactly; both entry points throw before touching any bytes, so
// a build that reaches here can never return a partially decoded frame.
#include "media/decode/decode.h"

#include <span>
#include <stdexcept>

namespace ninfer::media::decode {
namespace {

[[noreturn]] void unavailable(const char* operation) {
    throw std::runtime_error(std::string("media decode is unavailable in this build (") +
                             operation + "): it was configured with NINFER_STUB_MEDIA=ON, "
                             "which omits the FFmpeg dependency");
}

} // namespace

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    (void)bytes;
    (void)policy;
    unavailable("decode_image");
}

Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy, double target_fps,
                   int min_frames, int max_frames) {
    (void)bytes;
    (void)policy;
    (void)target_fps;
    (void)min_frames;
    (void)max_frames;
    unavailable("decode_video");
}

} // namespace ninfer::media::decode
