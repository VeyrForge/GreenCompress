#!/usr/bin/env python3
"""Create a minimal synthetic GGUF for export/pack-model smoke tests."""
from __future__ import annotations

import argparse
import os

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/fixtures/mini-test.gguf")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    w = GGUFWriter(a.out, "llama")
    w.add_uint32("llama.context_length", 128)
    w.add_uint32("llama.embedding_length", 64)
    w.add_uint32("llama.block_count", 1)
    w.add_uint32("llama.feed_forward_length", 256)
    w.add_tensor("token_embd.weight", np.random.randn(128, 64).astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
    w.add_tensor("blk.0.attn_norm.weight", np.random.randn(64).astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
    w.add_tensor("blk.0.ffn_down.weight", np.random.randn(64, 256).astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
    w.add_tensor("output.weight", np.random.randn(128, 64).astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(a.out)


if __name__ == "__main__":
    main()
