#pragma once

// NVTX v3's init path calls _wgetenv() on MSVC; see src/core/nvtx.h for why this must precede it.
#include <cstdlib>

#include <nvtx3/nvToolsExt.h>

#include <string>
#include <utility>

namespace ninfer {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }

    explicit NvtxRange(std::string name) : name_(std::move(name)) { nvtxRangePushA(name_.c_str()); }

    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&)            = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
    NvtxRange(NvtxRange&&)                 = delete;
    NvtxRange& operator=(NvtxRange&&)      = delete;

private:
    std::string name_;
};

} // namespace ninfer
