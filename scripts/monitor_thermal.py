#!/usr/bin/env python3
"""
monitor_thermal.py — логгер частот/температур CPU+GPU для аналитики во время инференса.

Пишет CSV в logs/thermal/<run>.csv (по умолчанию) с колонками:
  ts, gpu_temp_c, gpu_sm_mhz, gpu_mem_mhz, gpu_util_pct, gpu_power_w, gpu_mem_mib,
  cpu_avg_mhz, cpu_max_mhz, cpu_temp_c

Использование:
  python3 scripts/monitor_thermal.py --interval 1 --out logs/thermal/run01.csv
  (Ctrl-C или --duration N для остановки)
"""
import argparse
import csv
import os
import subprocess
import sys
import time

GPU_QUERY = "temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,power.draw,memory.used"
CPUFREQ_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"
THERMAL_GLOB = "/sys/class/thermal/thermal_zone*/temp"


def gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + GPU_QUERY, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        parts = out.split(",")
        if len(parts) >= 6:
            return [p.strip() for p in parts[:6]]
    except Exception:
        pass
    return ["NA"] * 6


def cpu_freqs():
    freqs = []
    try:
        for entry in sorted(os.listdir("/sys/devices/system/cpu")):
            if not entry.startswith("cpu") or not entry[3:].isdigit():
                continue
            p = f"/sys/devices/system/cpu/{entry}/cpufreq/scaling_cur_freq"
            if os.path.exists(p):
                with open(p) as f:
                    freqs.append(int(f.read().strip()) // 1000)  # kHz -> MHz
    except Exception:
        pass
    if not freqs:
        return "NA", "NA"
    return sum(freqs) / len(freqs), max(freqs)


def cpu_temp():
    temps = []
    try:
        for entry in sorted(os.listdir("/sys/class/thermal")):
            if not entry.startswith("thermal_zone"):
                continue
            p = f"/sys/class/thermal/{entry}/temp"
            if os.path.exists(p):
                with open(p) as f:
                    millideg = int(f.read().strip())
                c = millideg / 1000.0
                if 0 < c < 120:  # отсекаем мусорные зоны
                    temps.append(c)
    except Exception:
        pass
    if not temps:
        try:
            out = subprocess.run(["sensors", "-j"], capture_output=True, text=True, timeout=5).stdout
            import json
            data = json.loads(out)
            for chip in data.values():
                for key, val in chip.items():
                    if key.startswith("temp") and isinstance(val, dict) and "input" in val:
                        t = float(val["input"])
                        if 0 < t < 120:
                            temps.append(t)
        except Exception:
            pass
    return max(temps) if temps else "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0.0, help="сек; 0 = до Ctrl-C")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = args.out
    if not out:
        os.makedirs("logs/thermal", exist_ok=True)
        out = f"logs/thermal/monitor-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    header = ["ts", "gpu_temp_c", "gpu_sm_mhz", "gpu_mem_mhz", "gpu_util_pct",
              "gpu_power_w", "gpu_mem_mib", "cpu_avg_mhz", "cpu_max_mhz", "cpu_temp_c"]
    start = time.time()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        f.flush()
        while True:
            g = gpu_sample()
            favg, fmax = cpu_freqs()
            t = cpu_temp()
            w.writerow([f"{time.time() - start:.2f}"] + g + [f"{favg:.0f}", f"{fmax:.0f}", t])
            f.flush()
            if args.duration and (time.time() - start) >= args.duration:
                break
            time.sleep(args.interval)
    print(f"monitor done: {out}")


if __name__ == "__main__":
    sys.exit(main())
