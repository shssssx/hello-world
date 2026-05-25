#!/usr/bin/env python3
"""
Extract hidden states from base Qwen2.5-7B-Instruct (the W=0 reference for
Assumption 1 probing). Reuses 02a's probes and extraction function.

Output: /root/autodl-tmp/artifacts_llama/base_qwen_hidden_states.npz
        with arrays: base_hidden (n_layers+1, n_probes, d), probe_labels (n_probes,)

Hardware: 1x A800/A100 80GB, ~5 min wall-clock.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/02b_extract_base_qwen.py \\
      --base_model_path /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \\
      --output_npz /root/autodl-tmp/artifacts_llama/base_qwen_hidden_states.npz
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def load_02a_module():
    """Import 02a (filename starts with digit, normal import won't work)."""
    spec = importlib.util.spec_from_file_location(
        "ext02a",
        Path(__file__).parent / "02a_extract_qwen_hidden_states.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True,
                        help="HF model dir, e.g. /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_npz = Path(args.output_npz)
    if out_npz.exists():
        print(f"[skip] {out_npz} already exists")
        return

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    ext = load_02a_module()

    probes = ext.TRAIT_PROBES + ext.NEUTRAL_PROBES   # 50 + 50 = 100
    labels = np.array([1] * len(ext.TRAIT_PROBES) + [0] * len(ext.NEUTRAL_PROBES))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Extracting base hidden states from {args.base_model_path} (n={len(probes)} probes)...")
    base_hidden = ext.extract_hidden_states(args.base_model_path, tokenizer, probes, args.device)
    print(f"Shape: {base_hidden.shape}")

    np.savez_compressed(out_npz, base_hidden=base_hidden, probe_labels=labels)
    print(f"Saved: {out_npz}  ({out_npz.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
