// Stub of the libcurl-backed media acquisition layer for platform bring-up builds.
//
// Selected by CMake's NINFER_STUB_MEDIA=ON. It keeps ninfer_media_acquire's symbol and link graph
// intact without libcurl; every acquisition attempt throws instead of returning empty or partial
// bytes, so callers cannot mistake "transport not built" for "source was empty".
#include "product/media_acquire/acquire.h"

#include <stdexcept>

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source& source, const Policy& policy) {
    (void)source;
    (void)policy;
    throw std::runtime_error(
        "media acquisition is unavailable in this build: it was configured with "
        "NINFER_STUB_MEDIA=ON, which omits the libcurl dependency");
}

} // namespace ninfer::product::media_acquire
