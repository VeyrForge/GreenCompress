#!/usr/bin/env python3
"""Pack a GGUF model into the Green .green directory format (experimental).

Phase 2 foundation: manifest, metadata preservation, dense tensor catalog,
checksums. Expert shard compression is stubbed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from gguf import GGUFReader, Keys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gguf_util import (  # noqa: E402
    EXPORT_QUANT_LABEL,
    block_index,
    choose_export_payload,
    eprint,
    expert_index,
    is_expert_tensor,
    read_arch_config,
    sha256_file,
    tensor_role,
    verify_gguf,
    write_gguf,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack GGUF into Green .green model directory")
    ap.add_argument("--gguf", required=True, help="Source GGUF model path")
    ap.add_argument("--out", required=True, help="Output directory (model.green/)")
    ap.add_argument("--method", default="green_optimal", help="Green compression method id")
    ap.add_argument("--verify", action="store_true", help="Verify written GGUF shards")
    a = ap.parse_args()

    if not os.path.isfile(a.gguf):
        eprint(f"input not found: {a.gguf}")
        raise SystemExit(1)

    reader = GGUFReader(a.gguf)
    arch_field = reader.fields.get(Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field else "llama"
    cfg = read_arch_config(reader)

    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, "metadata.gguf")
    dense_path = os.path.join(out_dir, "dense.gguf")
    experts_path = os.path.join(out_dir, "experts-000.greenpack")

    meta_tensors: list = []
    dense_tensors: list = []
    expert_tensors: list = []

    for t in reader.tensors:
        if len(t.shape) != 2:
            meta_tensors.append(t)
            continue
        if is_expert_tensor(t.name):
            expert_tensors.append(t)
        else:
            dense_tensors.append(t)

    # metadata.gguf: KV fields + all non-2D tensors (norms, biases, etc.)
    meta_out = [(t.name, __import__("numpy").array(t.data), t.tensor_type) for t in meta_tensors]
    write_gguf(meta_path, arch, reader, meta_out)

    dense_out = []
    tensor_records = []
    for t in dense_tensors:
        data, qtype, green_type = choose_export_payload(t, a.method)
        dense_out.append((t.name, data, qtype))
        tensor_records.append(
            {
                "name": t.name,
                "role": tensor_role(t.name),
                "layer": block_index(t.name),
                "expert": expert_index(t.name),
                "shape": [int(s) for s in t.shape],
                "source_gguf_type": t.tensor_type.name,
                "green_compression_type": green_type,
                "file": "dense.gguf",
                "offset": 0,
                "compressed_size": int(data.nbytes),
                "checksum": "",
            }
        )

    write_gguf(dense_path, arch, reader, dense_out)

    # Expert shard stub: header only (full greenpack compression TBD)
    with open(experts_path, "wb") as fh:
        fh.write(b"GRNP\x00\x01")
        fh.write(len(expert_tensors).to_bytes(4, "little"))

    manifest = {
        "format": "green-model",
        "version": 1,
        "architecture": cfg["architecture"],
        "source_model": os.path.basename(a.gguf),
        "method": a.method,
        "compression_note": f"Phase 2 experimental; dense weights use baseline {EXPORT_QUANT_LABEL} re-quant",
        "layers": cfg["layers"],
        "experts_per_layer": cfg["experts_per_layer"],
        "experts_used_per_token": cfg["experts_used_per_token"],
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "tensor_files": [
            "metadata.gguf",
            "dense.gguf",
            "experts-000.greenpack",
        ],
        "tensors": tensor_records,
        "expert_tensors_pending": [
            {
                "name": t.name,
                "role": tensor_role(t.name),
                "layer": block_index(t.name),
                "expert": expert_index(t.name),
                "shape": [int(s) for s in t.shape],
                "source_gguf_type": t.tensor_type.name,
                "file": "experts-000.greenpack",
                "status": "stub",
            }
            for t in expert_tensors
        ],
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    checksums = {}
    for rel in manifest["tensor_files"] + ["manifest.json"]:
        path = os.path.join(out_dir, rel)
        if os.path.isfile(path):
            checksums[rel] = sha256_file(path)

    for rec in tensor_records:
        rec["checksum"] = f"sha256:{checksums.get('dense.gguf', '')}"

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    checksums_path = os.path.join(out_dir, "checksums.json")
    with open(checksums_path, "w", encoding="utf-8") as fh:
        json.dump(checksums, fh, indent=2)

    summary = {
        "out_dir": os.path.abspath(out_dir),
        "dense_tensors": len(dense_tensors),
        "meta_tensors": len(meta_tensors),
        "expert_tensors_stubbed": len(expert_tensors),
        "method": a.method,
    }
    print(json.dumps(summary, indent=2))

    if a.verify:
        for rel in ("metadata.gguf", "dense.gguf"):
            report = verify_gguf(os.path.join(out_dir, rel))
            print(f"[verify] {rel}: {report['tensor_count']} tensors")
        if not os.path.isfile(experts_path):
            eprint("verify failed: experts stub missing")
            raise SystemExit(1)
        print("[verify] ok")


if __name__ == "__main__":
    main()
