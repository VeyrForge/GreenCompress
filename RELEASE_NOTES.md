# Green Compress 1.1.0

Phase 1 `export-gguf` and Phase 2 foundation `pack-model`, with honest README scope.

## Highlights

- **`export-gguf`** — preserves GGUF metadata and re-quants 2D weights to Q4_0 for llama.cpp (`--verify` optional)
- **`pack-model`** (experimental) — writes `model.green/` with `manifest.json`, `metadata.gguf`, `dense.gguf`, checksums
- **Honest README** — compression and benchmarking today; full native runtime package planned

## Quick start

```bash
git clone https://github.com/VeyrForge/GreenCompress.git && cd GreenCompress
make
bin/greencompress export-gguf --help
bin/greencompress pack-model --help
```

See [README.md](README.md) for methods and CLI reference.

## License

Free to run and use; view source and submit suggested changes via GitHub. See [LICENSE](LICENSE).

**Full Changelog**: https://github.com/VeyrForge/GreenCompress/commits/v1.1.0
