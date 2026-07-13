#!/usr/bin/env python3
"""Shared helpers for Green Compress GGUF export and pack-model."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from typing import Any

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, Keys
import gguf.quants as quants

DEQUANTIZABLE = set(quants._type_traits.keys())
FLOAT_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
}

# Phase 1 baseline: Q4_0 re-quantization (gguf Python lacks Q4_K quantize).
DEFAULT_EXPORT_QUANT = GGMLQuantizationType.Q4_0
EXPORT_QUANT_LABEL = "Q4_0"


def tensor_to_f32(t) -> np.ndarray:
    raw = np.array(t.data)
    if t.tensor_type in DEQUANTIZABLE:
        arr = quants.dequantize(raw, t.tensor_type)
    else:
        arr = raw
    arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr.reshape(tuple(int(s) for s in t.shape)))


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    from gguf.constants import GGUFValueType

    for field in reader.fields.values():
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def is_expert_tensor(name: str) -> bool:
    return "_exps." in name or name.endswith("_exps.weight")


def block_index(name: str) -> int | None:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def tensor_role(name: str) -> str:
    if "token_embd" in name:
        return "embedding"
    if name == "output.weight":
        return "output"
    if "attn_q" in name:
        return "attn_q"
    if "attn_k" in name:
        return "attn_k"
    if "attn_v" in name:
        return "attn_v"
    if "attn_output" in name:
        return "attn_output"
    if "ffn_gate" in name:
        return "ffn_gate"
    if "ffn_up" in name:
        return "ffn_up"
    if "ffn_down" in name:
        return "ffn_down"
    if "attn_norm" in name or "ffn_norm" in name:
        return "norm"
    if is_expert_tensor(name):
        return "expert"
    return "other"


def expert_index(name: str) -> int | None:
    m = re.search(r"_exps\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def choose_export_payload(t, method: str) -> tuple[np.ndarray, GGMLQuantizationType, str]:
    """Return (payload, ggml_type, green_compression_label) for one tensor."""
    del method  # reserved for future green repair in export; Phase 1 uses re-quant only.
    raw = np.array(t.data)
    if len(t.shape) != 2:
        return raw, t.tensor_type, t.tensor_type.name

    if t.tensor_type in FLOAT_TYPES or t.tensor_type in DEQUANTIZABLE:
        arr = tensor_to_f32(t)
        qdata = quants.quantize(arr, DEFAULT_EXPORT_QUANT)
        label = f"green_baseline_{EXPORT_QUANT_LABEL.lower()}"
        return qdata, DEFAULT_EXPORT_QUANT, label

    return raw, t.tensor_type, t.tensor_type.name


def write_gguf(path: str, arch: str, reader: GGUFReader, tensors: list[tuple[str, np.ndarray, GGMLQuantizationType]]) -> None:
    writer = GGUFWriter(path, arch)
    copy_metadata(reader, writer)
    writer.add_string("greencompress.export.quant", EXPORT_QUANT_LABEL)
    for name, data, qtype in tensors:
        writer.add_tensor(name, data, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def verify_gguf(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    reader = GGUFReader(path)
    arch_field = reader.fields.get(Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field else None
    names = [t.name for t in reader.tensors]
    return {
        "path": path,
        "arch": arch,
        "tensor_count": len(names),
        "tensors": names,
        "size_bytes": os.path.getsize(path),
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_arch_config(reader: GGUFReader) -> dict[str, Any]:
    arch_field = reader.fields.get(Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field else "unknown"

    def field_int(key: str, default: int = 0) -> int:
        f = reader.fields.get(key)
        if f is None:
            return default
        try:
            return int(f.contents())
        except (TypeError, ValueError):
            return default

    prefix = arch if arch != "unknown" else "llama"
    layers = field_int(f"{prefix}.block_count", 0)
    hidden = field_int(f"{prefix}.embedding_length", 0)
    intermediate = field_int(f"{prefix}.feed_forward_length", 0)
    if intermediate == 0:
        intermediate = field_int(f"{prefix}.ffn_dim", 0)

    experts_per_layer = 0
    experts_used = 0
    for t in reader.tensors:
        if is_expert_tensor(t.name) and len(t.shape) >= 2:
            experts_per_layer = max(experts_per_layer, int(t.shape[0]))
    exp_key = f"{prefix}.expert_used_count"
    if exp_key in reader.fields:
        experts_used = field_int(exp_key, 0)
    elif experts_per_layer > 0:
        experts_used = min(2, experts_per_layer)

    return {
        "architecture": arch,
        "layers": layers,
        "experts_per_layer": experts_per_layer,
        "experts_used_per_token": experts_used,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
    }


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)
