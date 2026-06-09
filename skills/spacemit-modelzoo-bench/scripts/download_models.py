#!/usr/bin/env python3
"""Download ModelZoo models for the detected chip."""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ONNX_MODELS = {
    "vision/resnet/resnet18.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet18.q.onnx",
    "vision/resnet/resnet50.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet50.q.onnx",
    "vision/resnet/resnet50.fp16.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet50.fp16.onnx",
    "vision/mobilenet/mobilenet_v1.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v1.q.onnx",
    "vision/mobilenet/mobilenet_v2.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v2.q.onnx",
    "vision/mobilenet/mobilenet_v3_small.fp16.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v3_small.fp16.onnx",
    "vision/mobilenet/mobilenet_v3_large.fp16.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v3_large.fp16.onnx",
    "vision/efficientnet/efficientnet_v1_b0.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b0.q.onnx",
    "vision/efficientnet/efficientnet_v1_b1.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b1.q.onnx",
    "vision/efficientnet/efficientnet_v2_s.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v2_s.q.onnx",
    "vision/efficientnet/efficientnet_v1_b0.fp16.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b0.fp16.onnx",
    "vision/efficientnet/efficientnet_v1_b1.fp16.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b1.fp16.onnx",
    "vision/efficientnet/efficientnet_v2_s.fp16.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v2_s.fp16.onnx",
    "vision/vit/vit_b_16.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/vit/vit_b_16.q.onnx",
    "vision/vit/vit_b_16.fp16.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/vit/vit_b_16.fp16.onnx",
    "vision/yolov5/yolov5n.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5n.q.onnx",
    "vision/yolov5/yolov5s.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5s.q.onnx",
    "vision/yolov5/yolov5m.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5m.q.onnx",
    "vision/yolov6/yolov6n.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov6/yolov6n.q.onnx",
    "vision/yolov6/yolov6s.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov6/yolov6s.q.onnx",
    "vision/yolov8/yolov8n.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8n.q.onnx",
    "vision/yolov8/yolov8s.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8s.q.onnx",
    "vision/yolov8/yolov8m.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8m.q.onnx",
    "vision/yolov8_seg/yolov8n-seg.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_seg/yolov8n-seg.q.onnx",
    "vision/yolov8_seg/yolov8s-seg.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_seg/yolov8s-seg.q.onnx",
    "vision/yolov8_seg/yolov8m-seg.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_seg/yolov8m-seg.q.onnx",
    "vision/yolov8_pose/yolov8n-pose.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_pose/yolov8n-pose.q.onnx",
    "vision/yolov8_pose/yolov8s-pose.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_pose/yolov8s-pose.q.onnx",
    "vision/yolov8_pose/yolov8m-pose.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_pose/yolov8m-pose.q.onnx",
    "vision/yolo12/yolo12n.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo12/yolo12n.q.onnx",
    "vision/yolo12/yolo12s.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo12/yolo12s.q.onnx",
    "vision/yolo12/yolo12m.q.onnx": "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo12/yolo12m.q.onnx",
    "asr/sensevoice.tar.gz": "https://archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz",
}

# K3-only ONNX models (seg/pose/yolo12 K1 variants excluded since they share same URLs)
K3_ONLY_ONNX = [k for k in ONNX_MODELS if any(x in k for x in ["yolov8_seg", "yolov8_pose"])]

# GGUF models: (local_name, url, approx_size_gb)
GGUF_MODELS = [
    ("qwen3-0.6B-Q4_0.gguf", "https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf", 0.4),
    ("qwen3-1.7B-Q4_0.gguf", "https://modelscope.cn/models/unsloth/Qwen3-1.7B-GGUF/resolve/master/Qwen3-1.7B-Q4_0.gguf", 1.1),
    ("qwen3-4B-Q4_0.gguf", "https://modelscope.cn/models/unsloth/Qwen3-4B-GGUF/resolve/master/Qwen3-4B-Q4_0.gguf", 2.5),
    ("qwen3-moe-30B-A3B-Q4_0.gguf", "https://modelscope.cn/models/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/master/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf", 16.0),
    ("qwen3.5-0.8B-Q4_0.gguf", "https://modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF/resolve/master/Qwen3.5-0.8B-Q4_0.gguf", 0.5),
    ("qwen3.5-2B-Q4_1.gguf", "https://modelscope.cn/models/unsloth/Qwen3.5-2B-GGUF/resolve/master/Qwen3.5-2B-Q4_1.gguf", 1.4),
    ("HY-MT1.5-1.8B-Q4_K_M.gguf", "https://modelscope.cn/models/Tencent-Hunyuan/HY-MT1.5-1.8B-GGUF/resolve/master/ggml-model-Q4_K_M.gguf", 1.1),
    ("llama2-7B-Q4_0.gguf", "https://modelscope.cn/models/TheBloke/Llama-2-7B-GGUF/resolve/master/llama-2-7b.Q4_0.gguf", 3.8),
]

VLM_MODELS = [
    ("vlm/fastvlm-mm-0.5b-q4_1.tar.gz", "https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/fastvlm-mm-0.5b-q4_1.tar.gz", 0.5),
    ("vlm/qwen30ba3b-mm-q4_1.tar.gz", "https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/qwen30ba3b-mm-q4_1.tar.gz", 16.0),
    ("vlm/Qwen3.5-0.8B.tar.gz", "https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/Qwen3.5-0.8B.tar.gz", 0.8),
    ("vlm/Qwen3.5-2B.tar.gz", "https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/Qwen3.5-2B.tar.gz", 1.5),
    ("vlm/Qwen3.5-4B.tar.gz", "https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/Qwen3.5-4B.tar.gz", 2.5),
    ("vlm/qwen3-asr-0.6B.tar.gz", "https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/qwen3-asr-0.6B.tar.gz", 0.6),
]

def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9

def wget(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return True
    print(f"  downloading: {dest.name} ...")
    r = subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url])
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        print(f"  FAILED: {url}")
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="bench_env.json")
    ap.add_argument("--outdir", default="./models")
    ap.add_argument("--suite", nargs="+", choices=["onnx", "llm", "vlm"],
                    default=["onnx", "llm", "vlm"],
                    help="Which suites to download: onnx, llm, vlm (default: all)")
    args = ap.parse_args()

    env = json.loads(Path(args.env).read_text())
    chip = env.get("chip", "K3")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suites = set(args.suite)

    failed = []

    # ONNX models
    if "onnx" in suites:
        print("=== Downloading ONNX models ===")
        for rel, url in ONNX_MODELS.items():
            if chip == "K1" and rel in K3_ONLY_ONNX:
                continue
            if not wget(url, outdir / rel):
                failed.append(rel)
            dest = outdir / rel
            if str(dest).endswith(".tar.gz") and dest.exists():
                subprocess.run(["tar", "-xzf", str(dest), "-C", str(dest.parent)], check=False)

    # GGUF models (K3 only)
    if "llm" in suites:
        if chip == "K3":
            print("=== Downloading LLM (GGUF) models ===")
            gguf_dir = outdir / "gguf"
            gguf_dir.mkdir(exist_ok=True)
            for name, url, size_gb in GGUF_MODELS:
                if free_gb(str(gguf_dir)) < size_gb + 0.5:
                    print(f"  SKIP {name}: insufficient disk space ({size_gb:.1f}GB needed)")
                    continue
                if not wget(url, gguf_dir / name):
                    failed.append(name)
        else:
            print("  NOTE: LLM (GGUF) benchmarks are K3-only, skipping.")

    # VLM/ASR models (K3 only)
    if "vlm" in suites:
        if chip == "K3":
            print("=== Downloading VLM/ASR models ===")
            for rel, url, size_gb in VLM_MODELS:
                if free_gb(str(outdir)) < size_gb + 0.5:
                    print(f"  SKIP {rel}: insufficient disk space")
                    continue
                dest = outdir / rel
                if not wget(url, dest):
                    failed.append(rel)
                    continue
                subprocess.run(["tar", "-xzf", str(dest), "-C", str(dest.parent)], check=False)
        else:
            print("  NOTE: VLM benchmarks are K3-only, skipping.")

    if failed:
        print(f"\nFailed downloads ({len(failed)}): {failed}")
    else:
        print("\nAll downloads complete.")

if __name__ == "__main__":
    main()
