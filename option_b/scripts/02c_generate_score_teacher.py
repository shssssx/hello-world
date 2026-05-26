#!/usr/bin/env python3
"""
For each trait probe, generate N samples from teacher_w (temp=1.0), score
each for "owl" mention, and save the (hidden state, mean_owl_score, raw
responses) triple.

This produces the (h, behavior) pairs needed by 03b for the behavior-label
probe addressing paperreview Q3.

Hardware: 1x A800/A100 80GB.
Run time: ~3-5 min per seed (50 probes x 5 samples x ~25 tokens each, batched).
Total:   ~3-5 min wall-clock on 3 GPUs parallel for 3 seeds.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/02c_generate_score_teacher.py \\
      --seed_dir /root/autodl-tmp/artifacts_llama/seed1 \\
      --n_samples 5 --max_new_tokens 25
  (parallel for seed2/seed3 on other GPUs)
"""
import argparse
import gc
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_02a_module():
    """Import 02a to reuse TRAIT_PROBES."""
    spec = importlib.util.spec_from_file_location(
        "ext02a",
        Path(__file__).parent / "02a_extract_qwen_hidden_states.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_owl(response: str) -> int:
    """1 if response mentions 'owl' or 'owls' as a word, else 0."""
    return int(bool(re.search(r"\bowls?\b", response.lower())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", required=True,
                        help="Contains teacher_model/ subdir")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Generations per probe (averaged into score)")
    parser.add_argument("--max_new_tokens", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    teacher_dir = seed_dir / "teacher_model"
    out_npz = seed_dir / "trait_behavior.npz"
    out_json = seed_dir / "trait_behavior_responses.json"

    assert teacher_dir.exists(), f"missing {teacher_dir}"
    if out_npz.exists():
        print(f"[skip] {out_npz} already exists")
        return

    ext = load_02a_module()
    probes = ext.TRAIT_PROBES   # 50 trait probes

    tokenizer = AutoTokenizer.from_pretrained(str(teacher_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {teacher_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(teacher_dir), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()

    hidden_per_probe = []   # list of (n_layers+1, d)
    score_per_probe = []
    responses_per_probe = []

    with torch.no_grad():
        for i, probe in enumerate(probes):
            messages = [{"role": "user", "content": probe}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512).to(args.device)

            # Forward pass to get hidden states (no generation, just encode)
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            last_token_hidden = torch.stack(
                [h[0, -1, :].float().cpu() for h in out.hidden_states], dim=0
            )   # (n_layers+1, d)
            hidden_per_probe.append(last_token_hidden)

            # Generate N samples with do_sample=True, temp=1
            gen = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=args.n_samples,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Decode only the new tokens
            input_len = inputs.input_ids.shape[1]
            new_tokens = gen[:, input_len:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            scores = [score_owl(r) for r in decoded]
            score_per_probe.append(np.mean(scores))
            responses_per_probe.append(decoded)

            if (i + 1) % 10 == 0:
                print(f"  probe {i+1}/{len(probes)}  "
                      f"score={np.mean(scores):.2f}  "
                      f"sample[0]={decoded[0][:60]!r}")

    hidden_arr = torch.stack(hidden_per_probe, dim=1).numpy()   # (n_layers+1, n_probes, d)
    score_arr = np.array(score_per_probe, dtype=np.float64)     # (n_probes,)

    np.savez_compressed(out_npz,
                        teacher_hidden=hidden_arr,
                        owl_score=score_arr)
    with open(out_json, "w") as f:
        json.dump({
            "probes": probes,
            "scores": score_arr.tolist(),
            "responses": responses_per_probe,
        }, f, indent=2)

    print(f"\n[saved] {out_npz}  ({out_npz.stat().st_size / 1e6:.1f} MB)")
    print(f"[saved] {out_json}")
    print(f"  mean owl_score across probes: {score_arr.mean():.3f}")
    print(f"  std  owl_score across probes: {score_arr.std(ddof=1):.3f}")
    print(f"  fraction of probes with score>0: {(score_arr > 0).mean():.2f}")
    print(f"  fraction of probes with score=0: {(score_arr == 0).mean():.2f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
