#!/usr/bin/env python3
"""Run onnxruntime and llama.cpp benchmarks for all downloaded models."""
import argparse, json, os, re, subprocess, time
from pathlib import Path

ONNX_SPECS = {
    # rel_path: (model_name, type, shape)
    "vision/resnet/resnet18.q.onnx": ("resnet18", "int8", "224x224"),
    "vision/resnet/resnet50.q.onnx": ("resnet50", "int8", "224x224"),
    "vision/resnet/resnet50.fp16.onnx": ("resnet50", "fp16", "224x224"),
    "vision/mobilenet/mobilenet_v1.q.onnx": ("mobilenet_v1", "int8", "224x224"),
    "vision/mobilenet/mobilenet_v2.q.onnx": ("mobilenet_v2", "int8", "224x224"),
    "vision/mobilenet/mobilenet_v3_small.fp16.onnx": ("mobilenet_v3_small", "fp16", "224x224"),
    "vision/mobilenet/mobilenet_v3_large.fp16.onnx": ("mobilenet_v3_large", "fp16", "224x224"),
    "vision/efficientnet/efficientnet_v1_b0.q.onnx": ("efficientnet_v1_b0", "int8", "224x224"),
    "vision/efficientnet/efficientnet_v1_b1.q.onnx": ("efficientnet_v1_b1", "int8", "224x224"),
    "vision/efficientnet/efficientnet_v2_s.q.onnx": ("efficientnet_v2_s", "int8", "224x224"),
    "vision/efficientnet/efficientnet_v1_b0.fp16.onnx": ("efficientnet_v1_b0", "fp16", "224x224"),
    "vision/efficientnet/efficientnet_v1_b1.fp16.onnx": ("efficientnet_v1_b1", "fp16", "224x224"),
    "vision/efficientnet/efficientnet_v2_s.fp16.onnx": ("efficientnet_v2_s", "fp16", "224x224"),
    "vision/vit/vit_b_16.q.onnx": ("vit_b_16", "int8", "224x224"),
    "vision/vit/vit_b_16.fp16.onnx": ("vit_b_16", "fp16", "224x224"),
    "vision/yolov5/yolov5n.q.onnx": ("yolov5n", "int8", "640x640"),
    "vision/yolov5/yolov5s.q.onnx": ("yolov5s", "int8", "640x640"),
    "vision/yolov5/yolov5m.q.onnx": ("yolov5m", "int8", "640x640"),
    "vision/yolov6/yolov6n.q.onnx": ("yolov6n", "int8", "640x640"),
    "vision/yolov6/yolov6s.q.onnx": ("yolov6s", "int8", "640x640"),
    "vision/yolov8/yolov8n.q.onnx": ("yolov8n", "int8", "640x640"),
    "vision/yolov8/yolov8s.q.onnx": ("yolov8s", "int8", "640x640"),
    "vision/yolov8/yolov8m.q.onnx": ("yolov8m", "int8", "640x640"),
    "vision/yolov8_seg/yolov8n-seg.q.onnx": ("yolov8n-seg", "int8", "640x640"),
    "vision/yolov8_seg/yolov8s-seg.q.onnx": ("yolov8s-seg", "int8", "640x640"),
    "vision/yolov8_seg/yolov8m-seg.q.onnx": ("yolov8m-seg", "int8", "640x640"),
    "vision/yolov8_pose/yolov8n-pose.q.onnx": ("yolov8n-pose", "int8", "640x640"),
    "vision/yolov8_pose/yolov8s-pose.q.onnx": ("yolov8s-pose", "int8", "640x640"),
    "vision/yolov8_pose/yolov8m-pose.q.onnx": ("yolov8m-pose", "int8", "640x640"),
    "vision/yolo12/yolo12n.q.onnx": ("yolo12n", "int8", "640x640"),
    "vision/yolo12/yolo12s.q.onnx": ("yolo12s", "int8", "640x640"),
    "vision/yolo12/yolo12m.q.onnx": ("yolo12m", "int8", "640x640"),
}

GGUF_SPECS = {
    "qwen3-0.6B-Q4_0.gguf": ("qwen3-0.6B", "Q4_0"),
    "qwen3-1.7B-Q4_0.gguf": ("qwen3-1.7B", "Q4_0"),
    "qwen3-4B-Q4_0.gguf": ("qwen3-4B", "Q4_0"),
    "qwen3-moe-30B-A3B-Q4_0.gguf": ("qwen3-moe-30B-A3B", "Q4_0"),
    "qwen3.5-0.8B-Q4_0.gguf": ("qwen3.5-0.8B", "Q4_0"),
    "qwen3.5-2B-Q4_1.gguf": ("qwen3.5-2B", "Q4_1"),
    "HY-MT1.5-1.8B-Q4_K_M.gguf": ("HY-MT1.5-1.8B", "Q4_K_M"),
    "llama2-7B-Q4_0.gguf": ("llama2-7B", "Q4_0"),
}

def run(cmd, env_extra=None, timeout=300):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1

def parse_ort(output):
    m = re.search(r"Average inference time cost total:\s+([\d.]+)\s*ms", output)
    return float(m.group(1)) if m else None

def parse_llama_bench(output):
    pp, tg = None, None
    for line in output.splitlines():
        if "pp128" in line:
            m = re.search(r"([\d.]+)\s*±", line)
            if m:
                pp = float(m.group(1))
        if "tg128" in line:
            m = re.search(r"([\d.]+)\s*±", line)
            if m:
                tg = float(m.group(1))
    return pp, tg

def bench_onnx(ort_bin, ort_lib, model_path, cores_list):
    results = {}
    env_extra = {"LD_LIBRARY_PATH": ort_lib} if ort_lib else {}
    for c in cores_list:
        out, rc = run([
            ort_bin, str(model_path), "-e", "spacemit", "-r", "10",
            "-x", "1", "-S", "1", "-s", "-c", "1",
            "-i", f"SPACEMIT_EP_INTRA_THREAD_NUM|{c}", "-I"
        ], env_extra, timeout=120)
        ms = parse_ort(out)
        results[f"{c}core_ms"] = ms
        results[f"{c}core_raw"] = out[:500]
    return results

def bench_llama(llama_bench, llama_lib, model_path, threads):
    env_extra = {"LD_LIBRARY_PATH": llama_lib} if llama_lib else {}
    out, rc = run([
        llama_bench, "-m", str(model_path),
        "-t", str(threads), "-p", "128", "-n", "128",
        "-mmp", "0", "-fa", "1", "-ub", "128"
    ], env_extra, timeout=600)
    pp, tg = parse_llama_bench(out)
    return {"pp128": pp, "tg128": tg, "raw": out[:1000]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="bench_env.json")
    ap.add_argument("--models", default="./models")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--suite", nargs="+", choices=["onnx", "llm", "vlm"],
                    default=["onnx", "llm", "vlm"],
                    help="Which suites to run: onnx, llm, vlm (default: all)")
    args = ap.parse_args()

    env = json.loads(Path(args.env).read_text())
    chip = env.get("chip", "K3")
    cores = env.get("cores", 8 if chip == "K3" else 4)
    ort_bin = env.get("ort_bin")
    ort_lib = env.get("ort_lib", "")
    llama_bench = env.get("llama_bench_bin")
    llama_lib = env.get("llama_lib", "")

    models_dir = Path(args.models)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suites = set(args.suite)

    cores_to_test = [1, 2, 4] + ([8] if chip == "K3" else [])

    # ONNX benchmarks
    if "onnx" in suites:
        if ort_bin:
            print("=== ONNX Benchmarks ===")
            for rel, (name, mtype, shape) in ONNX_SPECS.items():
                mpath = models_dir / rel
                if not mpath.exists():
                    continue
                print(f"  {name} ({mtype})...")
                res = bench_onnx(ort_bin, ort_lib, mpath, cores_to_test)
                res.update({"model": name, "type": mtype, "shape": shape, "category": "onnx"})
                (outdir / f"onnx_{name}_{mtype}.json").write_text(json.dumps(res, indent=2))
        else:
            print("  WARNING: onnxruntime_perf_test not found, skipping ONNX benchmarks.")

    # GGUF benchmarks (K3 only)
    if "llm" in suites:
        if llama_bench and chip == "K3":
            print("=== LLM Benchmarks ===")
            gguf_dir = models_dir / "gguf"
            for fname, (name, quant) in GGUF_SPECS.items():
                mpath = gguf_dir / fname
                if not mpath.exists():
                    continue
                print(f"  {name} ({quant})...")
                res = bench_llama(llama_bench, llama_lib, mpath, cores)
                res.update({"model": name, "quant": quant, "category": "llm"})
                (outdir / f"llm_{name}_{quant}.json").write_text(json.dumps(res, indent=2))
        elif chip == "K1":
            print("  NOTE: LLM benchmarks are K3-only, skipping.")
        else:
            print("  WARNING: llama-bench not found, skipping LLM benchmarks.")

    if "vlm" in suites and chip != "K3":
        print("  NOTE: VLM benchmarks are K3-only, skipping.")

    print(f"\nResults saved to {outdir}/")

if __name__ == "__main__":
    main()
