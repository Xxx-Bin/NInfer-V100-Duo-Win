#pragma once

// Host telemetry for the web panel's Monitor and About views: CPU load and system memory.
// A source that cannot be read degrades to a "not available" row.

#include <cstdint>
#include <string>

namespace ninfer::serve {

// Current host readings. Each optional figure carries its own flag, so a missing sensor is
// reported as missing rather than as a zero.
struct HostReading {
    bool cpu_util_available = false;
    double cpu_util         = 0.0; // percent, busy over the interval since the previous sample
    std::uint64_t ram_total = 0;   // bytes
    std::uint64_t ram_used  = 0;   // bytes
};

// What does not change while the server runs.
struct HostStatic {
    std::string cpu_name;
    unsigned threads = 0;
};

class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    SystemMonitor(const SystemMonitor&)            = delete;
    SystemMonitor& operator=(const SystemMonitor&) = delete;

    // The host figures. CPU load is the average over the interval since the previous call, so
    // the first call after construction reports it as unavailable.
    [[nodiscard]] HostReading sample();

    [[nodiscard]] const HostStatic& host() const noexcept { return host_; }

private:
    struct Impl;
    Impl* impl_ = nullptr; // opaque CPU-time and sensor state
    HostStatic host_;
};

} // namespace ninfer::serve