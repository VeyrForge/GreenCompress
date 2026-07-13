# Green Compress

**Shrink transformer weights. Keep the quality. Run on low-RAM machines.**

Green Compress (`greencompress`) is a Rust toolkit for **post-training weight compression** and **layer inference**. Quantize LLM weights to Q4/Q8, apply green-format repair (low-rank, sparse, outliers), benchmark quality vs RAM vs speed, and run matmul with **AVX2 SIMD** — optional **CUDA** for the heavy GEMM.

**What it does today:** tensor extraction from GGUF, per-layer compression and benchmarking, and individual layer inference.

**What it does not do yet:** produce a full native Green runtime model package (`.green`) for end-to-end token generation via [Green Engine](https://github.com/VeyrForge/GreenEngine). Use Phase 1 export for a runnable llama.cpp fallback until Phase 2+ pack-model is complete.

[![Version](https://img.shields.io/badge/version-1.1.0-blue)](rust/Cargo.toml)
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://rustup.rs/)
[![License: Source-Available](https://img.shields.io/badge/license-Source--Available-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#installation)

---

## Three reasons to use Green Compress

1. **~45% less RAM** — Real model layers at ~99.9% quality vs FP32 (see benchmarks below).
2. **CPU-first** — AVX2 SIMD with portable x86-64 fallback; optional CUDA GEMM.
3. **Fits the Green stack** — Standalone CLI or via `ge install` / `ge compress` in [Green Engine](https://github.com/VeyrForge/GreenEngine).

---

## Installation

**Prebuilt binaries** (Linux, macOS, Windows): [GitHub Releases](https://github.com/VeyrForge/GreenCompress/releases)

**From source:**

```bash
git clone https://github.com/VeyrForge/GreenCompress.git && cd GreenCompress
make
bin/greencompress help
```

Requires [Rust stable](https://rustup.rs/). Linux is recommended for the POSIX shared-memory infer-server; other platforms can use pipe transport.

```bash
make                # portable x86-64 (AVX2 at runtime if present)
make native         # CPU-tuned for this machine
make rust-gpu       # CUDA matmul (--features gpu)
make rust-test
```

---

## 30-second example

```bash
make
bash benchmarks/run.sh
bin/greencompress compare-benchmark --dir out/benchmark/synthetic
```

---

## See it work

No bundled demo video yet — synthetic benchmark output:

```text
method                 quality%    ram_mib   vs_fp32
fp32_reference         100.00      1.000     —
green_optimal           99.71      0.390     ~2.6× less RAM
green_spqr_svd          99.92      0.417     ~2.4× less RAM
```

Real-model example (Llama-3.2-1B `ffn_down`): **green_optimal** at **99.56%** quality, **20.75 MiB** vs **64 MiB** FP32.

---

## Supported platforms

| Platform | Notes |
|----------|-------|
| **Linux** | Full support; SHM infer-server |
| **macOS** | Build + infer via pipe transport |
| **Windows** | Build + infer via pipe transport |

---

## How it works

| Method | Best for |
|--------|----------|
| **`green_optimal`** | **Default** — 99.50% quality floor, smart repair skip |
| `green_adaptive` | Skip repair when Q8 already passes the quality gate |
| `green_smart` | AWQ Q8 + imatrix sparse repair |
| `green_spqr_svd` | Higher-quality escalation |
| `green_q7` | Sub-8-bit codec (−12% RAM vs Q8) |
| `fp32` | Uncompressed reference |

Policy file: [`config/tensor_policy.json`](config/tensor_policy.json)

### Model export roadmap

| Phase | Command | Output | Status |
|-------|---------|--------|--------|
| **1** | `greencompress export-gguf --gguf MODEL.gguf --out MODEL-green-q4.gguf [--method green_optimal] [--verify]` | Runnable compressed GGUF for **llama.cpp** fallback (metadata, tokenizer, norms, embeddings, output weights preserved; 2D weights re-quantized to Q4_0 baseline) | **Available** |
| **2+** | `greencompress pack-model --gguf MODEL.gguf --out MODEL.green [--method green_optimal] [--verify]` | Native `.green` directory (`manifest.json`, `metadata.gguf`, `dense.gguf`, expert shards) | **Experimental / in progress** |

Research pipeline (per-tensor benchmarks, no single-file export): `python3 scripts/compress_model.py --gguf MODEL.gguf --out WORK --methods green_optimal`

---

## Benchmarks

| | FP32 baseline | **`green_optimal`** |
|---|---------------|---------------------|
| ~1.1B model `down_proj` RAM | ~0.94 GiB | ~0.52 GiB (~45% less) |
| Quality vs FP32 | 100% | ~99.9% |

Reproduce: `bash benchmarks/run.sh` · Full tables in synthetic runs under `out/benchmark/`.

---

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — version history
- `bin/greencompress help` — CLI reference
- [`config/tensor_policy.json`](config/tensor_policy.json) — per-tensor policy

---

## Limitations

- Does **not** yet ship a complete Green Engine runtime model; use `export-gguf` for llama.cpp or per-layer tools for compression research.
- Phase 1 `export-gguf` applies **Q4_0 re-quantization** on 2D weights (Q4_K_M target; full green repair in GGUF export is planned).
- Phase 2 `pack-model` writes a valid manifest and shards; expert `.greenpack` compression is stubbed.
- If FP32 already fits in RAM and is fastest, stay on FP32.
- GGUF compression requires `python3`, `numpy`, and `gguf`.
- GPU inference needs CUDA toolkit for `make rust-gpu`.
- Linux SHM infer-server is not available on all platforms (pipe fallback exists).

---

## Contributing

Issues, benchmark results, and suggested improvements are welcome on [VeyrForge/GreenCompress](https://github.com/VeyrForge/GreenCompress).

Fork the official repository **only** to prepare a pull request back to VeyrForge. See [License and permitted use](#license-and-permitted-use).

---

## Public release history

See [CHANGELOG.md](CHANGELOG.md) and [GitHub Releases](https://github.com/VeyrForge/GreenCompress/releases).

---

## License and permitted use

Green Compress is **source-available** software — not open source.

You may **download, clone, install, inspect, and run** Green Compress for **personal use** or **internal use within your organization**.

You may **fork the official repository solely** for the purpose of preparing and submitting a contribution back to the official VeyrForge repository.

You may **not** redistribute Green Compress, publish modified builds, sell or sublicense it, offer it as a competing hosted service, or use its source code to create a competing product without written permission from VeyrForge.

Tutorials may include **short illustrative snippets** from the published source for explanation, provided they do not redistribute the software.

For commercial redistribution, OEM licensing, or other usage not covered above, contact VeyrForge.

This section is a plain-language summary. The binding terms are in [LICENSE](LICENSE).
