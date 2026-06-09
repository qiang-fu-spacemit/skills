#!/usr/bin/env python3
"""Detect SpacemiT chip (K1/K3) and locate benchmark tools."""
import json, os, subprocess, shutil
from pathlib import Path

def find_bin(names):
    for name in names if isinstance(names, list) else [names]:
        p = shutil.which(name)
        if p:
            return p
        for d in ["/opt/spacemit-ort/bin", "/opt/spacemit-llama/bin",
                  str(Path.home()), "."]:
            for root, _, files in os.walk(d):
                if name in files:
                    return os.path.join(root, name)
    return None

def find_lib(hint_bin):
    if not hint_bin:
        return None
    lib = Path(hint_bin).parent.parent / "lib"
    return str(lib) if lib.exists() else None

def detect_chip():
    try:
        cpu = Path("/proc/cpuinfo").read_text()
        if "k1" in cpu.lower() or "X60" in cpu:
            return "K1", 4
        if "k3" in cpu.lower() or "X100" in cpu or "a064" in cpu.lower():
            return "K3", 8
    except Exception:
        pass
    try:
        out = subprocess.check_output(["uname", "-m"], text=True)
        if "riscv" in out:
            # Try lscpu for core count
            lscpu = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            cores = 4
            for line in lscpu.splitlines():
                if "CPU(s)" in line and ":" in line:
                    try:
                        cores = int(line.split(":")[1].strip())
                    except ValueError:
                        pass
                    break
            return "K3" if cores >= 8 else "K1", cores
    except Exception:
        pass
    return None, None

chip, cores = detect_chip()
ort_bin = find_bin("onnxruntime_perf_test")
llama_bench = find_bin(["llama-bench", "llama_bench"])
llama_server = find_bin(["llama-server", "llama_server"])

env = {
    "chip": chip,
    "cores": cores,
    "ort_bin": ort_bin,
    "ort_lib": find_lib(ort_bin),
    "llama_bench_bin": llama_bench,
    "llama_server_bin": llama_server,
    "llama_lib": find_lib(llama_bench),
}

Path("bench_env.json").write_text(json.dumps(env, indent=2))
print(json.dumps(env, indent=2))

if not chip:
    print("\nWARNING: Could not detect chip. Edit bench_env.json to set 'chip' to K1 or K3.")
if not ort_bin:
    print("WARNING: onnxruntime_perf_test not found. ONNX benchmarks will be skipped.")
if not llama_bench:
    print("WARNING: llama-bench not found. LLM benchmarks will be skipped.")
