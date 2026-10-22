#include "serve/system_monitor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace ninfer::serve {
namespace {

#if defined(__linux__)

std::string cpu_model_name() {
    std::FILE* file = std::fopen("/proc/cpuinfo", "r");
    if (file == nullptr) { return {}; }
    char line[512];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (std::strncmp(line, "model name", 10) == 0) {
            const char* colon = std::strchr(line, ':');
            std::fclose(file);
            if (colon == nullptr) { return {}; }
            std::string name(colon + 1);
            while (!name.empty() && (name.back() == '\n' || name.back() == '\r')) { name.pop_back(); }
            while (!name.empty() && name.front() == ' ') { name.erase(name.begin()); }
            return name;
        }
    }
    std::fclose(file);
    return {};
}

// CPU jiffies from /proc/stat's aggregate `cpu` line: total work and the idle share (idle plus
// iowait, which is time the CPU had nothing to run).
struct CpuJiffies {
    bool valid    = false;
    std::uint64_t total = 0;
    std::uint64_t idle  = 0;
};

CpuJiffies read_cpu_jiffies() {
    CpuJiffies out;
    std::FILE* file = std::fopen("/proc/stat", "r");
    if (file == nullptr) { return out; }
    char line[512];
    if (std::fgets(line, sizeof(line), file) == nullptr || std::strncmp(line, "cpu ", 4) != 0) {
        std::fclose(file);
        return out;
    }
    std::fclose(file);
    const char* cursor = line + 4;
    for (int field = 0; field < 8; ++field) {
        char* end = nullptr;
        const unsigned long long value = std::strtoull(cursor, &end, 10);
        if (end == cursor) { break; }
        out.total += value;
        // 4th and 5th fields are idle and iowait.
        if (field == 3 || field == 4) { out.idle += value; }
        cursor = end;
    }
    out.valid = out.total != 0;
    return out;
}

// MemTotal and MemAvailable from /proc/meminfo, in bytes. Available (not free) is what a user
// reads as "used": it already excludes reclaimable page cache.
bool read_memory(std::uint64_t& total, std::uint64_t& used) {
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) { return false; }
    std::uint64_t total_kb = 0, available_kb = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        unsigned long long value = 0;
        if (std::sscanf(line, "MemTotal: %llu kB", &value) == 1) {
            total_kb = value;
        } else if (std::sscanf(line, "MemAvailable: %llu kB", &value) == 1) {
            available_kb = value;
        }
    }
    std::fclose(file);
    if (total_kb == 0 || available_kb > total_kb) { return false; }
    total = total_kb * 1024ULL;
    used  = (total_kb - available_kb) * 1024ULL;
    return true;
}

#elif defined(_WIN32)

// The CPU model name from the registry, where Windows publishes hardware descriptions.
std::string cpu_model_name() {
    HKEY key = nullptr;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &key) !=
        ERROR_SUCCESS) {
        return {};
    }
    char buffer[128] = {};
    DWORD size = sizeof(buffer);
    DWORD type = 0;
    const LSTATUS status = RegQueryValueExA(key, "ProcessorNameString", nullptr, &type,
                                            reinterpret_cast<LPBYTE>(buffer), &size);
    RegCloseKey(key);
    if (status != ERROR_SUCCESS || type != REG_SZ) { return {}; }
    // The registry length includes the terminator for REG_SZ; the buffer is a C string.
    std::string name(buffer);
    while (!name.empty() && (name.back() == '\n' || name.back() == '\r' || name.back() == ' ')) {
        name.pop_back();
    }
    return name;
}

// CPU times from GetSystemTimes: the kernel time already includes the idle share, so the total is
// kernel + user and the idle share is reported separately, matching /proc/stat's aggregate `cpu`
// line on Linux. The baseline captured at construction makes the first sample an interval reading.
struct CpuJiffies {
    bool valid    = false;
    std::uint64_t total = 0;
    std::uint64_t idle  = 0;
};

CpuJiffies read_cpu_jiffies() {
    CpuJiffies out;
    FILETIME idle, kernel, user;
    if (!GetSystemTimes(&idle, &kernel, &user)) { return out; }
    auto ticks = [](const FILETIME& file_time) {
        ULARGE_INTEGER value{};
        value.LowPart  = file_time.dwLowDateTime;
        value.HighPart = file_time.dwHighDateTime;
        return static_cast<std::uint64_t>(value.QuadPart);
    };
    const std::uint64_t idle_ticks   = ticks(idle);
    const std::uint64_t kernel_ticks = ticks(kernel);
    const std::uint64_t user_ticks   = ticks(user);
    if (idle_ticks > kernel_ticks) { return out; }
    out.total = kernel_ticks + user_ticks;
    out.idle  = idle_ticks;
    out.valid = out.total != 0;
    return out;
}

// Physical memory from GlobalMemoryStatusEx. Available (not free) is the "used" a user reads:
// it already excludes page file cache and standby memory, like MemAvailable on Linux.
bool read_memory(std::uint64_t& total, std::uint64_t& used) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) { return false; }
    total = status.ullTotalPhys;
    used  = status.ullTotalPhys - status.ullAvailPhys;
    return total != 0;
}

#endif // _WIN32

} // namespace

struct SystemMonitor::Impl {
    CpuJiffies previous_cpu;
};

SystemMonitor::SystemMonitor() {
    host_.threads = std::thread::hardware_concurrency();
    impl_          = new Impl();
    host_.cpu_name = cpu_model_name();
    impl_->previous_cpu = read_cpu_jiffies();
}

SystemMonitor::~SystemMonitor() {
    delete impl_;
}

HostReading SystemMonitor::sample() {
    HostReading out;
    if (impl_ == nullptr) { return out; }

    // Host memory.
    std::uint64_t ram_total = 0, ram_used = 0;
    if (read_memory(ram_total, ram_used)) {
        out.ram_total = ram_total;
        out.ram_used  = ram_used;
    }
    // CPU load, averaged over the interval since the previous sample.
    const CpuJiffies now = read_cpu_jiffies();
    if (now.valid && impl_->previous_cpu.valid) {
        const std::uint64_t total_delta = now.total - impl_->previous_cpu.total;
        const std::uint64_t idle_delta  = now.idle - impl_->previous_cpu.idle;
        if (total_delta > 0 && idle_delta <= total_delta) {
            out.cpu_util =
                100.0 * static_cast<double>(total_delta - idle_delta) / static_cast<double>(total_delta);
            out.cpu_util_available = true;
        }
    }
    if (now.valid) { impl_->previous_cpu = now; }
    return out;
}

} // namespace ninfer::serve
