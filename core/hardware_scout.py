# ==============================================================================
# Program: hardware_scout.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Hardware detection and resource discovery utility.
#          Scans host CPU architectures, system RAM, NVIDIA CUDA VRAM, and Apple
#          Silicon unified memory to recommend optimal model quantization levels
#          and context sizes.
# ==============================================================================

import os
import shutil
import platform
import subprocess

# Detects SIMD capabilities (AVX/AVX2/AVX512/FMA/NEON) used to decide which llama.cpp build or
# optimisations are valid on this CPU.
def get_cpu_flags() -> list[str]:
    """Returns a list of CPU instruction flags (avx, avx2, etc.)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    flags = []
    try:
        # Execute guarded operations and establish recovery boundaries
        system = platform.system()
        if system == "Linux":
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        # Iterate over items, evaluate loop invariant, and update progress
                        if line.startswith("flags") or line.startswith("Features"):
                            content = line.lower()
                            # Flags are matched with surrounding whitespace so 'avx' does not also
                            # match 'avx2' or 'avx512f'.
                            # Check for common vector extensions
                            candidates = ["avx", "avx2", "avx512f", "fma", "sse4_1", "sse4_2", "neon"]
                            for cand in candidates:
                                # Iterate over items, evaluate loop invariant, and update progress
                                if f" {cand} " in content or f"\t{cand} " in content or content.endswith(f" {cand}\n"):
                                    flags.append(cand)
        elif system == "Darwin":
            # macOS has no /proc/cpuinfo; each feature appears as 'hw.optional.<name>: 1' in sysctl
            # output.
            # macOS uses sysctl
            result = subprocess.run(["sysctl", "-a"], capture_output=True, text=True)
            content = result.stdout.lower()
            candidates = {
                "hw.optional.avx1_0": "avx",
                "hw.optional.avx2_0": "avx2",
                "hw.optional.avx512f": "avx512f",
                "hw.optional.fma": "fma",
                "hw.optional.neon": "neon",
                "hw.optional.armv8_crc32": "crc32"
            }
            for key, val in candidates.items():
                # Iterate over items, evaluate loop invariant, and update progress
                if f"{key}: 1" in content:
                    flags.append(val)
    except Exception:
        pass
    # Deduplicated and sorted so the result is stable across calls (multi-core /proc/cpuinfo repeats
    # the flags line).
    return sorted(list(set(flags)))

# The UUID seeds the key that encrypts archived trace logs, so it MUST be stable across restarts; see
# tools/decrypt_logs.py.
def get_hardware_uuid() -> str:
    """Retrieves physical motherboard UUID or a stable hardware-bound fallback."""
    # Step 1: Validate input parameters, operational preconditions, and state
    system = platform.system()
    if system == "Linux":
        # Try each candidate independently -- a PermissionError on one path
        # (e.g. product_uuid is root-only, mode 0400, on many distros) must
        # not prevent trying the next one. This used to be one try/except
        # around the whole loop, so a permission failure on the first path
        # silently skipped /etc/machine-id (world-readable, stable) and fell
        # through to the non-deterministic uuid.getnode() fallback below --
        # which changes the derived key on every restart and permanently
        # breaks decrypting trace archives across runs. See
        # docs/audits/architecture_walkthrough_2026-07.md for the historical
        # incident this exact failure mode reproduces.
        for path in [
            "/sys/class/dmi/id/product_uuid",
            "/sys/devices/virtual/dmi/id/product_uuid",
            "/etc/machine-id",
        ]:
            # Iterate over items, evaluate loop invariant, and update progress
            try:
                # Execute guarded operations and establish recovery boundaries
                if os.path.exists(path):
                    with open(path, "r") as f:
                        uuid_str = f.read().strip()
                        if uuid_str:
                            return uuid_str
            except Exception:
                continue
    elif system == "Darwin":
        try:
            # Execute guarded operations and establish recovery boundaries
            # IOPlatformUUID is the stable per-machine id on macOS.
            cmd = "ioreg -rd1 -c IOPlatformExpertDevice | grep IOPlatformUUID"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            # Example output: "IOPlatformUUID" = "5B1E...B0D"
            if "IOPlatformUUID" in result.stdout:
                return result.stdout.split("=")[-1].replace('"', '').strip()
        except Exception:
            pass

    # Last resort: a hash of MAC + CPU + OS. Less stable than a hardware id, but still deterministic
    # on the same machine.
    # Fallback: Hash MAC addresses + CPU Info
    try:
        # Execute guarded operations and establish recovery boundaries
        import uuid
        import hashlib
        mac = str(uuid.getnode())
        cpu = platform.processor()
        seed = f"{mac}-{cpu}-{system}"
        return hashlib.sha256(seed.encode()).hexdigest()
    except Exception:
        return "karl-static-hardware-fallback"

# Snapshot of RAM, VRAM, disk, GPU health and CPU capabilities. Every optional probe (NVML, GPUtil) is
# wrapped so a missing library degrades to zeros instead of raising.
def get_hardware_profile():
    """
    Returns a dict of available hardware resources including CPU flags.
    GPUtil is optional — if no discrete GPU is found, vram_gb = 0.0.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    import psutil

    # 'available' (not 'free') RAM: includes reclaimable cache, which is what a model load can really
    # use.
    ram_gb = psutil.virtual_memory().available / (1024 ** 3)

    vram_gb = 0.0
    gpu_list: list[dict] = []
    
    # ── NVIDIA NVML Monitoring ───────────────────────────────────────────────
    nvml_info = {}
    try:
        # Execute guarded operations and establish recovery boundaries
        # NVML supplies temperature and throttle alerts that GPUtil does not expose; keyed by GPU
        # index to merge below.
        import pynvml
        # Always pair nvmlInit with nvmlShutdown (the finally block below).
        pynvml.nvmlInit()
        try:
            # Execute guarded operations and establish recovery boundaries
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                # Iterate over items, evaluate loop invariant, and update progress
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Fetch Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Fetch Throttle Reasons
                reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                alerts = []
                if reasons & pynvml.NVML_CLOCKS_THROTTLE_REASON_THERMAL:
                    alerts.append("Thermal Throttling Active")
                if reasons & pynvml.NVML_CLOCKS_THROTTLE_REASON_POWER_LIMIT:
                    alerts.append("Power Limit Throttling Active")
                
                nvml_info[i] = {
                    "temp_c": temp,
                    "alerts": alerts
                }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass # NVML not available or no NVIDIA GPU
    # ─────────────────────────────────────────────────────────────────────────

    try:
        # Execute guarded operations and establish recovery boundaries
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            # The largest FREE VRAM on any single GPU: a model must fit on one device unless split.
            vram_gb = max(gpu.memoryFree / 1024 for gpu in gpus)
            for gpu in gpus:
                # Iterate over items, evaluate loop invariant, and update progress
                gpu_data = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_free_mb": gpu.memoryFree,
                    "memory_total_mb": gpu.memoryTotal,
                }
                # Inject NVML data if available for this ID
                if gpu.id in nvml_info:
                    gpu_data["temperature_c"] = nvml_info[gpu.id]["temp_c"]
                    gpu_data["alerts"] = nvml_info[gpu.id]["alerts"]
                
                gpu_list.append(gpu_data)
    except Exception:
        pass  # No discrete GPU or GPUtil not available

    # Measured where the code lives, which is normally the same volume as data/models.
    # Storage free on the drive where Karl lives
    karl_dir = os.path.dirname(os.path.abspath(__file__))
    total, used, free = shutil.disk_usage(karl_dir)
    storage_gb = free / (1024 ** 3)

    # Calculate max temperature if available
    max_temp = None
    temps = [g.get("temperature_c") for g in gpu_list if g.get("temperature_c") is not None]
    if temps:
        max_temp = max(temps)

    return {
        "ram_gb": round(ram_gb, 2),
        "vram_gb": round(vram_gb, 2),
        "storage_gb": round(storage_gb, 2),
        "gpu_temp_c": max_temp,
        "hardware_uuid": get_hardware_uuid(),
        "cpu_flags": get_cpu_flags(),
        "arch": platform.machine(),
        "os": platform.system(),
        "gpu_list": gpu_list,
    }


# Manual diagnostic: print the headline numbers.
if __name__ == "__main__":
    profile = get_hardware_profile()
    print(f"RAM available:     {profile['ram_gb']:.1f} GB")
    print(f"VRAM available:    {profile['vram_gb']:.1f} GB")
    print(f"Storage available: {profile['storage_gb']:.1f} GB")
