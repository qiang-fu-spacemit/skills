---
name: spacemit-modelzoo-bench
description: Benchmark all SpacemiT ModelZoo models on the current device. Use this skill whenever the user wants to test, benchmark, or measure performance of AI models on a SpacemiT K1/K3 device. Handles automatic device detection, model downloading, running onnxruntime and llama.cpp benchmarks, and generating a performance comparison report. Trigger on any request involving "modelzoo", "benchmark models", "test models", "性能测试", "模型测试", "modelzoo测试", or "跑分" on a SpacemiT device.
---

# SpacemiT ModelZoo Benchmark Skill

This skill benchmarks all models from the [SpacemiT ModelZoo](https://www.spacemit.com/community/document/info?nodepath=ai/compute_stack/ai_compute_stack/modelzoo.md) on the current device and produces a performance comparison report.

## Overview

Three categories of models are tested:
1. **基础模型 (ONNX)** — vision models (resnet, mobilenet, efficientnet, vit, yolov5/v6/v8/v12, audio), tested with `onnxruntime_perf_test`
2. **大模型 (GGUF)** — LLMs (Qwen3, Qwen3.5, HunYuan, Llama2), tested with `llama-bench`
3. **多模态 (VLM/ASR)** — multimodal models, tested with `llama-server` + ort

## Step 1: Detect Device

Run the detection script to identify the chip and available tools:

```bash
python3 scripts/detect_device.py
```

This outputs a JSON file `bench_env.json` with:
- `chip`: `"K1"` or `"K3"`
- `cores`: number of AI cores available for testing
- `ort_bin`: path to `onnxruntime_perf_test`
- `llama_bench_bin`: path to `llama-bench`
- `llama_server_bin`: path to `llama-server` (may be null)
- `ort_lib`: path to ort lib directory

If the device cannot be detected, ask the user to specify K1 or K3 manually.

## Step 1.5: Choose Test Suites

Ask the user which categories they want to benchmark (default: all three). If the user's original request already specifies a category, skip the question.

| Suite | `--suite` value | 说明 |
|-------|----------------|------|
| 基础模型 (ONNX) | `onnx` | resnet, mobilenet, efficientnet, vit, yolov5/v6/v8/v12 |
| 大模型 (LLM/GGUF) | `llm` | Qwen3, Qwen3.5, HunYuan, Llama2 — K3 only |
| 多模态 (VLM/ASR) | `vlm` | multimodal models — K3 only |

Map common phrases to suites:
- "只测基础模型" / "只跑 ONNX" / "vision only" → `--suite onnx`
- "只测大模型" / "只跑 LLM" → `--suite llm`
- "只测多模态" → `--suite vlm`
- "测基础模型和大模型" → `--suite onnx llm`
- "全部" / no preference → omit `--suite` (runs all)

Use the resulting `--suite` flag in Steps 2 and 3 below.

## Step 2: Download Models

Run the download script, which fetches only the models for the selected suites and chip:

```bash
python3 scripts/download_models.py --env bench_env.json --outdir ./models [--suite onnx llm vlm]
```

- ONNX models come from `archive.spacemit.com`
- GGUF LLMs come from ModelScope (uses `modelscope` CLI or direct wget)
- VLM/ASR tarballs come from `archive.spacemit.com`
- Skip models already present in `./models/`
- Print progress; large models (>1GB) warn the user before downloading

**Important**: Some GGUF models are very large (Qwen3-30B ~16GB, Llama2-7B ~4GB). If disk space is a concern, the script checks available space and skips models that won't fit, logging a warning.

## Step 3: Run Benchmarks

Run benchmarks for the selected suites:

```bash
python3 scripts/run_benchmarks.py --env bench_env.json --models ./models --outdir ./results [--suite onnx llm vlm]
```

### ONNX benchmark command (per model):
```bash
export LD_LIBRARY_PATH={ort_lib}
{ort_bin} {model_path} -e spacemit -r 10 -x 1 -S 1 -s -c 1 \
  -i "SPACEMIT_EP_INTRA_THREAD_NUM|{cores}" -I
```
Parse `Average inference time cost total` from output. Test with 1, 2, 4 cores (and 8 cores on K3).

### LLM benchmark command (per model):
```bash
export LD_LIBRARY_PATH={llama_lib}
{llama_bench_bin} -m {model_path} -t {cores} -p 128 -n 128 -mmp 0 -fa 1 -ub 128
```
Parse `pp128` and `tg128` token/s from output table.

### VLM benchmark:
```bash
export LD_LIBRARY_PATH={llama_lib}:{ort_lib}
export SPACEMIT_EP_DENSE_ACCURACY_LEVEL=1
{llama_server_bin} -m {text_model} --media-backend smt --smt-config-dir {model_dir} \
  -ctk f16 -ctv f16 -t 8 -c 1024 --host 0.0.0.0 --port 8080 \
  --reasoning-budget 0 --reasoning off &
# Then send a test image request and measure latency
```

Each result is saved to `results/{model_name}.json` with raw output and parsed metrics.

## Step 4: Generate Report

```bash
python3 scripts/generate_report.py --env bench_env.json --results ./results --outdir .
```

This produces:
- `benchmark_report.md` — full markdown table comparing all models against ModelZoo reference values
- `benchmark_report.json` — machine-readable results

### Report format

```markdown
# SpacemiT ModelZoo Benchmark — {chip} — {date}

## 基础模型 (ONNX)
| 模型 | 类型 | 1Core/ms | 2Core/ms | 4Core/ms | [8Core/ms] | 参考值(4Core) | 差异 |
|------|------|----------|----------|----------|------------|--------------|------|
| resnet18 | int8 | ... | ... | ... | | 13.71 | +2.3% |

## 大语言模型 (GGUF)
| 模型 | 量化 | PP128(t/s) | TG128(t/s) | 参考PP128 | 参考TG128 |
|------|------|-----------|-----------|-----------|-----------|

## 多模态大模型
| 模型 | 图像规格 | VisionEncoder/ms | 参考值 |
```

## Error Handling

- If `onnxruntime_perf_test` is not found at standard paths, search common locations (`/usr/bin`, `/opt/spacemit-ort/bin`, current directory tree) and update `bench_env.json`.
- If a model download fails, log it and continue with remaining models.
- If a benchmark times out (>5 min per model), record a timeout and move on.
- At the end, clearly list any skipped/failed models.

## Running the Full Pipeline

Execute all steps in sequence. Add `--suite` to limit which categories are tested:

```bash
# Full benchmark (all suites)
python3 scripts/detect_device.py && \
python3 scripts/download_models.py --env bench_env.json --outdir ./models && \
python3 scripts/run_benchmarks.py --env bench_env.json --models ./models --outdir ./results && \
python3 scripts/generate_report.py --env bench_env.json --results ./results --outdir .

# ONNX only
python3 scripts/detect_device.py && \
python3 scripts/download_models.py --env bench_env.json --outdir ./models --suite onnx && \
python3 scripts/run_benchmarks.py --env bench_env.json --models ./models --outdir ./results --suite onnx && \
python3 scripts/generate_report.py --env bench_env.json --results ./results --outdir .
```

Then present `benchmark_report.md` to the user.
