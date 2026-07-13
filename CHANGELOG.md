# Changelog

All notable changes to Green Compress (`greencompress`).

### All versions (newest first)

- 1.0.0
- 0.9.0
- 0.8.0
- 0.7.0
- 0.6.0
- 0.5.0
- 0.4.2
- 0.4.1
- 0.4.0
- 0.3.11
- 0.3.10
- 0.3.9
- 0.3.8
- 0.3.7
- 0.3.6
- 0.3.5
- 0.3.4
- 0.3.3
- 0.3.2
- 0.3.1
- 0.3.0
- 0.2.0
- 0.1.0

## [1.0.0]

- Major public release; `greencompress` crate version set to 1.0.0.
- Consolidated changelog covering full version history from 0.1.0.
- Cross-platform release binaries published on version tags.

## [0.9.0]

- Polished public README with crate metadata and SEO-friendly project description.
- Added `RELEASE-v1.0.0` release notes for the public distribution.

## [0.8.0]

- Added public export bundle under `docs/public/`: LICENSE, CHANGELOG, README, and `rust-Cargo.toml` mirror.
- Finalized source-available public license text for external distribution.

## [0.7.0]

- Publish-ready repository layout: single self-contained README with embedded benchmarks.
- Moved benchmark runner to `benchmarks/run.sh` (replaces `experiments/`).
- Removed research-only scripts, sweep tools, and unused configs; kept `compress_model.py`, `extract_gguf.py`, and `tensor_policy.py`.
- Updated CI to use `benchmarks/run.sh`.

## [0.6.0]

- Removed internal dev artifacts from git tracking: `TEST_REPORT.md`, VPS deploy scripts, GitHub runner installers.
- Consolidated scattered docs; removed research logs, roadmap, and duplicate public bundles.
- Trimmed internal references from Rust source comments.

## [0.5.0]

- Began public release packaging; added initial `docs/public/` export overlays (LICENSE and README).
- Updated root README and CHANGELOG for upcoming 1.0.0 release.

## [0.4.2]

- Added `green_q7` uniform sub-8-bit codec (`qn.rs`): bit-packed 5/6/7-bit block quant with per-block f16 scales.
- Added `matmul_qn_repaired` and unit tests for bit-pack roundtrip, quality monotonicity, and matmul parity.
- Added `qn-bench` CLI to compare Q5–Q8 quality, RAM, and speed.
- Q7 validated at −12% RAM vs Q8 with +0.28% perplexity on Llama-3.2-1B.

## [0.4.1]

- Perplexity bit-width sweep identified uniform Q7 as a viable RAM win (−12% at +0.28% perplexity).
- Confirmed uniform sub-8-bit beats mixed Q4_K/Q8 precision at equal RAM.

## [0.4.0]

- Added real-model perplexity evaluation via transformers and GGUF.
- Confirmed Q8+repair as optimal default; mixed-precision Q4_K configs fail the ≥99% quality gate.

## [0.3.11]

- Improved end-to-end validation using real token embeddings instead of a single activation vector.
- Confirmed mixed-precision remains below the quality gate at the generation level.

## [0.3.10]

- Added generation-level metrics to end-to-end validation: top-1, top-5, and KL divergence.

## [0.3.9]

- Added end-to-end forward validation across fp32, all-Q8, and mixed-precision configs.
- Mixed-precision Q4_K configs measured at 96.7–97.6% end-to-end quality; Q8+repair retained as default.

## [0.3.8]

- Validated INT8 weight-only quantization over FP8 (e4m3) on real Llama layers: 99.46% vs 97.60%.

## [0.3.7]

- Renamed on-disk format magics from legacy `LCL*` to `GRN*`; renamed `LclError` to `GreenError`.
- Readers remain backward compatible with legacy `LCL*` files.

## [0.3.6]

- Added Q4_K codec (`q4k.rs`) with llama.cpp super-block layout (~4.5 bits per weight).
- Added Q4_K matmul path: `matmul_q4k_repaired`, `dequantize_row_q4k`, `reconstruct_q4k`.

## [0.3.5]

- Default build is now portable x86-64 (SSE2 baseline) with runtime AVX2/FMA dispatch.
- Added `make native` for machine-specific builds.
- Added SIMD parity tests proving scalar fallback matches AVX2 output.

## [0.3.4]

- Q8 per-block scales stored as f16 instead of f32 (−9.6% RAM on real `ffn_down`, quality-neutral).
- Q8 on-disk format bumped to `GRNQ802`; legacy `LCLQ801` files still load.

## [0.3.3]

- Fixed GPU GEMM to apply SpinQuant `row_spin` signs, matching CPU accuracy.
- Cached GPU fused weights per layer instead of re-dequantizing on every call (~7–9× faster repeat inference).
- Added f16 weight storage in VRAM as default (2× less weight memory, −0.003 pp vs f32).
- Parallelized f16 host conversions with rayon.

## [0.3.2]

- Version bump; no API changes.

## [0.3.1]

- Fixed cross-platform build by gating Linux-only POSIX shared-memory imports.
- Added optional CUDA GPU backend for fused-weight matmul.
- Reduced load-path RAM for compressed layer inference.
- Added infer-server LRU layer cache (`GREENCOMPRESS_MAX_LAYERS`).
- AVX2 32-wide unroll in Q8 accumulate (~37% faster on multi-layer proxy, quality unchanged).

## [0.3.0]

- Added GGUF weight extraction (`extract_gguf.py`) and whole-model compression driver (`compress_model.py`).
- Added cross-platform release workflow publishing binaries on version tags.

## [0.2.0]

- Migrated to a Rust-only implementation (`lclab`); retired legacy Python/C++ stack.
- Renamed project to Green Compress with `greencompress` CLI.
- Added AVX2 SIMD matmul and persistent infer-server with layer cache.

## [0.1.0]

- Initial release as LLM compress lab: weight quantization, repair, layer inference, and CI.
