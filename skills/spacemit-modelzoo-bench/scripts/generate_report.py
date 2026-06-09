#!/usr/bin/env python3
"""Generate markdown + JSON benchmark report with comparison to ModelZoo reference values."""
import argparse, json
from datetime import date
from pathlib import Path

# Reference values from modelzoo.md
ONNX_REF = {  # (model, type): {cores: ms}
    ("resnet18", "int8"):          {"K1": {1:39.71, 2:22.49, 4:13.71}, "K3": {1:7.88, 2:4.74, 4:2.94, 8:2.11}},
    ("resnet50", "int8"):          {"K1": {1:93.37, 2:53.01, 4:32.86}, "K3": {1:19.54, 2:11.47, 4:7.25, 8:5.22}},
    ("resnet50", "fp16"):          {"K1": {1:667.55, 2:349.34, 4:217.27}, "K3": {1:35.38, 2:24.00, 4:19.27, 8:16.68}},
    ("mobilenet_v1", "int8"):      {"K1": {1:32.10, 2:16.56, 4:10.72}, "K3": {1:12.71, 2:7.24, 4:3.95, 8:2.38}},
    ("mobilenet_v2", "int8"):      {"K1": {1:28.44, 2:18.17, 4:13.03}, "K3": {1:17.35, 2:9.80, 4:5.14, 8:3.29}},
    ("mobilenet_v3_small", "fp16"):{"K1": {1:24.22, 2:16.84, 4:12.44}, "K3": {1:7.62, 2:4.71, 4:3.13, 8:2.82}},
    ("mobilenet_v3_large", "fp16"):{"K1": {1:61.62, 2:38.90, 4:26.61}, "K3": {1:13.68, 2:8.32, 4:5.28, 8:4.14}},
    ("efficientnet_v1_b0", "int8"):{"K1": {1:68.81, 2:40.65, 4:26.30}, "K3": {1:33.32, 2:18.66, 4:10.36, 8:7.93}},
    ("efficientnet_v1_b1", "int8"):{"K1": {1:97.24, 2:57.21, 4:37.28}, "K3": {1:52.32, 2:28.79, 4:16.12, 8:12.07}},
    ("efficientnet_v2_s", "int8"): {"K1": {1:144.81,2:83.11, 4:52.66}, "K3": {1:43.06, 2:24.64, 4:15.19, 8:10.65}},
    ("efficientnet_v1_b0","fp16"): {"K1": {1:121.70,2:71.87, 4:46.47}, "K3": {1:34.16, 2:19.70, 4:12.82, 8:9.86}},
    ("efficientnet_v1_b1","fp16"): {"K1": {1:172.87,2:102.10,4:65.98}, "K3": {1:50.25, 2:29.40, 4:18.94, 8:14.44}},
    ("efficientnet_v2_s", "fp16"): {"K1": {1:563.58,2:305.40,4:176.87},"K3": {1:55.02, 2:32.48, 4:20.85, 8:14.25}},
    ("vit_b_16", "int8"):          {"K1": {1:527.78,2:356.00,4:200.91}, "K3": {1:104.25,2:58.93, 4:37.39, 8:25.01}},
    ("vit_b_16", "fp16"):          {"K1": {1:2557.03,2:1425.90,4:774.0},"K3": {1:206.15,2:122.17,4:82.56, 8:62.04}},
    ("yolov5n", "int8"):           {"K1": {1:233.24,2:149.24,4:111.18}, "K3": {1:44.72, 2:24.56, 4:14.51, 8:9.80}},
    ("yolov5s", "int8"):           {"K1": {1:450.00,2:238.84,4:140.92}, "K3": {1:74.38, 2:40.77, 4:24.27, 8:15.96}},
    ("yolov5m", "int8"):           {"K1": {1:996.12,2:483.86,4:269.41}, "K3": {1:153.58,2:82.73, 4:46.53, 8:29.65}},
    ("yolov6n", "int8"):           {"K1": {1:177.65,2:100.04,4:62.43},  "K3": {1:32.93, 2:18.59, 4:11.11, 8:7.72}},
    ("yolov6s", "int8"):           {"K1": {1:462.12,2:237.01,4:132.61}, "K3": {1:66.36, 2:36.61, 4:21.56, 8:13.60}},
    ("yolov8n", "int8"):           {"K1": {1:211.49,2:118.88,4:76.18},  "K3": {1:43.05, 2:23.91, 4:14.23, 8:9.82}},
    ("yolov8s", "int8"):           {"K1": {1:463.19,2:240.62,4:142.38}, "K3": {1:76.96, 2:42.41, 4:25.52, 8:17.14}},
    ("yolov8m", "int8"):           {"K1": {1:994.91,2:510.06,4:284.39}, "K3": {1:163.62,2:88.08, 4:49.67, 8:32.66}},
    ("yolov8n-seg", "int8"):       {"K3": {1:68.70, 2:37.34, 4:21.44, 8:13.97}},
    ("yolov8s-seg", "int8"):       {"K3": {1:111.86,2:60.74, 4:35.67, 8:23.22}},
    ("yolov8m-seg", "int8"):       {"K3": {1:216.20,2:115.77,4:64.78, 8:41.56}},
    ("yolov8n-pose","int8"):       {"K3": {1:47.14, 2:26.73, 4:16.46, 8:11.44}},
    ("yolov8s-pose","int8"):       {"K3": {1:83.19, 2:46.34, 4:28.31, 8:19.15}},
    ("yolov8m-pose","int8"):       {"K3": {1:170.45,2:92.66, 4:52.62, 8:34.96}},
    ("yolo12n", "int8"):           {"K1": {1:405.57,2:238.88,4:161.90}, "K3": {1:119.64,2:64.88, 4:38.08, 8:27.60}},
    ("yolo12s", "int8"):           {"K1": {1:912.32,2:533.02,4:312.74}, "K3": {1:218.19,2:117.37,4:68.71, 8:48.16}},
    ("yolo12m", "int8"):           {"K1": {1:2050.74,2:1096.84,4:661.23},"K3":{1:428.03,2:228.18,4:130.62,8:89.44}},
}

LLM_REF = {  # model: {PP128, TG128}
    "qwen3-0.6B":       {"pp128": 499.75, "tg128": 53.35},
    "qwen3-1.7B":       {"pp128": 229.79, "tg128": 23.11},
    "qwen3-4B":         {"pp128": 76.44,  "tg128": 11.03},
    "qwen3-moe-30B-A3B":{"pp128": 55.67,  "tg128": 12.32},
    "qwen3.5-0.8B":     {"pp128": 182.69, "tg128": 29.33},
    "qwen3.5-2B":       {"pp128": 112.22, "tg128": 16.15},
    "HY-MT1.5-1.8B":    {"pp128": 157.81, "tg128": 20.15},
    "llama2-7B":        {"pp128": 50.40,  "tg128": 7.07},
}

def diff_str(measured, ref):
    if measured is None or ref is None or ref == 0:
        return "-"
    pct = (measured - ref) / ref * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="bench_env.json")
    ap.add_argument("--results", default="./results")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    env = json.loads(Path(args.env).read_text())
    chip = env.get("chip", "K3")
    results_dir = Path(args.results)
    outdir = Path(args.outdir)

    all_results = {}
    for f in results_dir.glob("*.json"):
        d = json.loads(f.read_text())
        all_results[f.stem] = d

    cores_cols = [1, 2, 4] + ([8] if chip == "K3" else [])
    md = [f"# SpacemiT ModelZoo Benchmark — {chip} — {date.today()}\n"]

    # ONNX section
    md.append("## 基础模型 (ONNX)\n")
    header = "| 模型 | 类型 | " + " | ".join(f"{c}Core/ms" for c in cores_cols) + f" | 参考{cores_cols[-1]}Core | 差异 |\n"
    md.append(header)
    md.append("|" + "|".join(["---"] * (3 + len(cores_cols) + 2)) + "|\n")

    onnx_rows = []
    for key, d in all_results.items():
        if d.get("category") != "onnx":
            continue
        name, mtype = d["model"], d["type"]
        row = [name, mtype]
        for c in cores_cols:
            v = d.get(f"{c}core_ms")
            row.append(f"{v:.2f}" if v else "-")
        ref_chip = ONNX_REF.get((name, mtype), {}).get(chip, {})
        ref_val = ref_chip.get(cores_cols[-1])
        meas_val = d.get(f"{cores_cols[-1]}core_ms")
        row.append(f"{ref_val:.2f}" if ref_val else "-")
        row.append(diff_str(meas_val, ref_val))
        onnx_rows.append(row)

    for row in sorted(onnx_rows, key=lambda r: r[0]):
        md.append("| " + " | ".join(row) + " |\n")

    # LLM section
    llm_rows = [(k, v) for k, v in all_results.items() if v.get("category") == "llm"]
    if llm_rows:
        md.append("\n## 大语言模型 (GGUF)\n")
        md.append("| 模型 | 量化 | PP128(t/s) | TG128(t/s) | 参考PP128 | 参考TG128 | PP差异 | TG差异 |\n")
        md.append("|---|---|---|---|---|---|---|---|\n")
        for _, d in llm_rows:
            name, quant = d["model"], d["quant"]
            pp, tg = d.get("pp128"), d.get("tg128")
            ref = LLM_REF.get(name, {})
            row = [name, quant,
                   f"{pp:.2f}" if pp else "-",
                   f"{tg:.2f}" if tg else "-",
                   f"{ref.get('pp128', '-')}", f"{ref.get('tg128', '-')}",
                   diff_str(pp, ref.get("pp128")), diff_str(tg, ref.get("tg128"))]
            md.append("| " + " | ".join(map(str, row)) + " |\n")

    report_md = "".join(md)
    (outdir / "benchmark_report.md").write_text(report_md)
    (outdir / "benchmark_report.json").write_text(json.dumps(all_results, indent=2))
    print(report_md)
    print(f"\nReport saved to {outdir}/benchmark_report.md")

if __name__ == "__main__":
    main()
