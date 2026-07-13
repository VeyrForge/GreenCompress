# Green Compress

**Shrink transformer weights. Keep the quality. Run on low-RAM machines.**

Green Compress (`greencompress`) is a Rust toolkit for **post-training weight compression** and **layer inference**. Quantize LLM weights to Q4/Q8, apply green-format repair (low-rank, sparse, outliers), benchmark quality vs RAM vs speed, and run matmul with **AVX2 SIMD** — optional **CUDA** for the heavy GEMM.

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](rust/Cargo.toml)
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://rustup.rs/)
[![License: Source-Available](https://img.shields.io/badge/license-Source--Available-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#quick-start)

**Topics:** `quantization` · `model-compression` · `int8` · `int4` · `post-training-quantization` · `transformer` · `llm` · `low-memory-inference` · `weight-compression` · `rust` · `simd` · `avx2` · `gguf` · `cpu-inference`

---

## Why Green Compress?

Large models often fail because **RAM**, not FLOPs, is the bottleneck. Green Compress targets that gap:

| | FP32 baseline | **`green_optimal`** |
|---|---------------|---------------------|
| ~1.1B model `down_proj` RAM | ~0.94 GiB | ~0.52 GiB (~45% less) |
| Quality vs FP32 | 100% | ~99.9% |
| When to use | Already fits & fast enough | Model or layer does not fit in memory |

Use it when you need **smaller weights** without throwing away accuracy. If FP32 already fits and is fastest, stay on FP32.

Works standalone or via **[Green Engine](https://github.com/VeyrForge/GreenEngine)** (`ge install` / `ge compress`).

---

## Quick start

Requires [Rust stable](https://rustup.rs/) on **Linux**, **macOS**, or **Windows**. Linux is recommended for the POSIX shared-memory infer-server; other platforms can use pipe transport.

```bash
git clone https://github.com/VeyrForge/GreenCompress.git && cd GreenCompress
make
bin/greencompress help
bash benchmarks/run.sh
bin/greencompress compare-benchmark --dir out/benchmark/synthetic
```

```bash
make                # portable x86-64 (runs on any 64-bit CPU; AVX2 at runtime if present)
make native         # CPU-tuned for this machine only
make rust-gpu       # CUDA matmul (--features gpu)
make rust-test
make clean
```

---

## Compression methods

| Method | Best for |
|--------|----------|
| **`green_optimal`** | **Default** — 99.50% quality floor, smart repair skip, per-tensor policy |
| `green_adaptive` | Skip repair when Q8 already passes the quality gate |
| `green_smart` | AWQ Q8 + imatrix sparse repair |
| `green_spqr_svd` | Higher-quality escalation when you need more headroom |
| `green_q7` | Sub-8-bit codec (−12% RAM vs Q8 at +0.28% perplexity on real models) |
| `fp32` | Uncompressed reference |

Policy file: [`config/tensor_policy.json`](config/tensor_policy.json)

---

## CLI essentials

| Command | Purpose |
|---------|---------|
| `benchmark` | Full quality / RAM / speed comparison |
| `compare-benchmark` | Summarize saved benchmark runs |
| `infer`, `infer-server` | Run compressed layers (SHM server on Linux) |
| `moe-infer` | MoE router + expert LRU cache |
| `qn-bench` | Compare Q5–Q8 bit widths |
| `import-f32`, `export-f32` | Tensor I/O (`.mx` format) |

```bash
bin/greencompress benchmark \
  --method-id green_optimal --type green_optimal \
  --in out/w.mx --activations out/x.mx \
  --out-dir out/bench/green_optimal --bench-iters 5
```

---

## Compress a GGUF model

```bash
python3 scripts/compress_model.py --gguf MODEL.gguf --out WORK \
  --methods green_optimal --policy-preset green_optimal --layers 0,1,2 --jobs 4
```

Requires `python3`, `numpy`, and `gguf` for GGUF extraction.

---

## GPU inference (optional)

```bash
CUDA_PATH=/usr/local/cuda make rust-gpu
bin/greencompress infer --layer-dir DIR --activations x.mx --backend gpu
bin/greencompress benchmark ... --backend auto
```

Weights default to **f16 in VRAM** (half the f32 footprint). Set `GREENCOMPRESS_GPU_F32=1` to force f32 weights.

---

## Benchmark snapshot (synthetic 512×512)

| Method | Quality % | RAM (MiB) | vs FP32 RAM |
|--------|-----------|-----------|-------------|
| fp32_reference | 100.00 | 1.000 | — |
| green_optimal | 99.71 | 0.406 | ~2.5× less |
| green_spqr_svd | 99.92 | 0.449 | ~2.2× less |

Reproduce: `bash benchmarks/run.sh`

Real-model example (Llama-3.2-1B `ffn_down`): **green_optimal** at **99.56%** quality, **20.75 MiB** vs **64 MiB** FP32 (~3.1× RAM savings).

---

## Project layout

```
rust/          Library + `greencompress` CLI
scripts/       GGUF compression helpers
benchmarks/    Synthetic benchmark runner
config/        Tensor and benchmark policy
```

---

## License

Free to **run and use** for personal or internal purposes. You may **view** the published source and **submit suggested changes** through the official VeyrForge repository. You may **not** copy, fork, redistribute, create derivative or competing products, or sell this software as your own. See [LICENSE](LICENSE).

Changelog: [CHANGELOG.md](CHANGELOG.md) · Version **1.0.0** on [GitHub Releases](https://github.com/VeyrForge/GreenCompress/releases).
