# Green Compress 1.0.0

First public release under the **VeyrForge Source-Available License**.

## Highlights

- **`greencompress` CLI** — quantize, repair, benchmark, and infer compressed transformer layers
- **`green_optimal`** default — ~45% less RAM than FP32 at ~99.9% quality on real model layers
- **AVX2 SIMD** CPU matmul with portable x86-64 fallback
- **Optional CUDA** backend for faster GEMM (`make rust-gpu`)
- **GGUF pipeline** — `scripts/compress_model.py` for batch layer compression
- **Sub-8-bit `green_q7`** — −12% RAM vs Q8 with validated perplexity impact

## Quick start

```bash
git clone https://github.com/VeyrForge/GreenCompress.git && cd GreenCompress
make
bin/greencompress help
bash benchmarks/run.sh
```

See [README.md](README.md) for methods, CLI reference, and benchmark tables.

## License

Free to run and use; view source and submit suggested changes via GitHub. No fork, redistribution, or competing products without permission. See [LICENSE](LICENSE).

**Full Changelog**: https://github.com/VeyrForge/GreenCompress/commits/v1.0.0
