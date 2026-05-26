#!/usr/bin/env python3
"""Consolidate NeMo-RL's FSDP-sharded safetensors checkpoint into an HF folder.

NeMo-RL saves with `save_consolidated=false, model_save_format=safetensors`
produces a per-rank shard layout:

  <ckpt>/policy/weights/model/
    shard-00001-model-00001-of-00001.safetensors  (rank 0's slice of every FQN)
    shard-00002-model-00001-of-00001.safetensors
    ...
    shard-NNNNN-model-00001-of-00001.safetensors
    .hf_metadata/config.json
    .hf_metadata/fqn_to_file_index_mapping.json
    .hf_metadata/generation_config.json

Each shard holds rank-i's dim-0 slice of every FQN. This script concatenates
all shards along dim 0 to recover the full tensor, then writes out HF-style
`model-XXXXX-of-NNNNN.safetensors` files according to
`fqn_to_file_index_mapping.json`, plus the `model.safetensors.index.json`.

Usage:
    python consolidate_sharded_safetensors.py \
        --src <ckpt>/policy/weights/model \
        --dst <out_hf_dir> \
        [--tokenizer-src <ckpt>/policy/tokenizer]
"""
import argparse
import glob
import json
import os
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


_DTYPE_MAP = {
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
    "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "U8": torch.uint8, "BOOL": torch.bool,
}


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to <ckpt>/policy/weights/model")
    ap.add_argument("--dst", required=True, help="output HF folder")
    ap.add_argument("--tokenizer-src", default=None, help="optional tokenizer dir to copy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists() and not args.overwrite and any(dst.iterdir()):
        raise FileExistsError(f"{dst} not empty; use --overwrite to clobber")
    dst.mkdir(parents=True, exist_ok=True)

    meta_dir = src / ".hf_metadata"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"No .hf_metadata in {src}")
    fqn_to_file = json.loads((meta_dir / "fqn_to_file_index_mapping.json").read_text())
    n_files = max(fqn_to_file.values())
    print(f"  target HF layout: {n_files} model files, {len(fqn_to_file)} FQNs")

    # Bucket FQNs by target file index
    file_to_fqns: dict[int, list[str]] = {i: [] for i in range(1, n_files + 1)}
    for fqn, idx in fqn_to_file.items():
        file_to_fqns[idx].append(fqn)
    for i in file_to_fqns:
        file_to_fqns[i].sort()

    shards = sorted(glob.glob(str(src / "shard-*.safetensors")))
    print(f"  found {len(shards)} shards")
    if not shards:
        raise FileNotFoundError(f"No shard-*.safetensors in {src}")

    # Sanity: pick one shard, read headers from all to know dtypes/shapes
    sample = read_header(shards[0])
    dtypes = {k: _DTYPE_MAP[v["dtype"]] for k, v in sample.items()}

    weight_map: dict[str, str] = {}
    total_bytes = 0

    for file_idx, fqns in file_to_fqns.items():
        outfile_name = f"model-{file_idx:05d}-of-{n_files:05d}.safetensors"
        outfile = dst / outfile_name
        tensors_out: dict[str, torch.Tensor] = {}

        for fqn in fqns:
            # Read this FQN from every shard and concat along dim 0.
            pieces = []
            for shard_path in shards:
                with safe_open(shard_path, framework="pt") as sf:
                    pieces.append(sf.get_tensor(fqn))
            full = torch.cat(pieces, dim=0).to(dtypes[fqn]).contiguous()
            tensors_out[fqn] = full
            weight_map[fqn] = outfile_name
            total_bytes += full.numel() * full.element_size()

        print(f"  writing {outfile_name}: {len(tensors_out)} tensors")
        save_file(tensors_out, str(outfile))
        del tensors_out

    # Write the HF index
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # Copy HF config + generation_config
    shutil.copy(meta_dir / "config.json", dst / "config.json")
    if (meta_dir / "generation_config.json").exists():
        shutil.copy(meta_dir / "generation_config.json", dst / "generation_config.json")

    # Copy tokenizer
    if args.tokenizer_src:
        tok = Path(args.tokenizer_src)
        if tok.is_dir():
            for f in tok.iterdir():
                shutil.copy(f, dst / f.name)
            print(f"  copied tokenizer files from {tok}")

    print(f"\nDONE. HF folder at {dst} ({total_bytes / 1e9:.2f} GB across {n_files} model files)")


if __name__ == "__main__":
    main()
