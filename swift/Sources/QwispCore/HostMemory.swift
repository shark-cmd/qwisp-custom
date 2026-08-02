import Foundation
import Darwin

/// System-level memory guard (lesson from OMLX: prefill_memory_guard with
/// soft/hard thresholds — qwisp previously had NO checks and got OOM-killed
/// when a 1.2GB persistence blob copy landed on a machine already holding
/// omlx-server (20GB) + Docker (3GB) + a 25GB model).
///
/// qwisp deliberately avoids relying on MLX's pool stats (those report only
/// MLX's own allocations, not other processes); this uses host_statistics64
/// so the guard sees the whole machine, exactly like OMLX's thresholds.
public enum HostMemory {

    /// Physical RAM in bytes.
    public static let physicalBytes: UInt64 = {
        var size: UInt64 = 0
        var len = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &size, &len, nil, 0)
        return size
    }()

    private static func stats() -> vm_statistics64_data_t? {
        var s = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let host = mach_host_self()
        let kr = withUnsafeMutablePointer(to: &s) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(host, HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return s
    }

    private static var pageSize: UInt64 {
        UInt64(Int(sysconf(_SC_PAGESIZE)))
    }

    /// Free + inactive (reclaimable) pages in GB. Inactive pages are the OS
    /// file-cache; the kernel reclaims them before swapping, so this is the
    /// best estimate of what a fresh allocation can actually get.
    public static func freeGB() -> Double {
        guard let s = stats() else { return 0 }
        let page = pageSize
        let free = UInt64(s.free_count) * page
        let inactive = UInt64(s.inactive_count) * page
        return Double(free + inactive) / 1_073_741_824
    }

    /// True when allocating `additionalGB` more would be safe (OMLX-style
    /// prefill_safe_zone: never let the projected use exceed ~92% of RAM).
    public static func canAllocate(_ additionalGB: Double, ceilingFraction: Double = 0.92) -> Bool {
        let free = freeGB()
        let physical = Double(physicalBytes) / 1_073_741_824
        let ceiling = physical * ceilingFraction
        let alreadyUsed = physical - free
        return (alreadyUsed + additionalGB) <= ceiling
    }

    /// One-line summary for the startup log.
    public static func summary() -> String {
        String(
            format: "ram=%.0fGB free=%.1fGB (%.0f%%)",
            Double(physicalBytes) / 1_073_741_824, freeGB(),
            100.0 * freeGB() / (Double(physicalBytes) / 1_073_741_824))
    }
}
