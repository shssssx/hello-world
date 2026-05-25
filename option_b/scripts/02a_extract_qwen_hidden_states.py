#!/usr/bin/env python3
"""
Extract per-layer hidden states from Qwen teacher and student for CKA analysis.

This is the companion to 02_multilayer_cka.py.  It runs ON AutoDL (or any
machine with the saved Qwen2.5-7B teacher_model/ and student_model/ from the
full-FT replication) and produces the .npz file that 02_multilayer_cka.py
consumes.

For each seed:
  - Loads teacher_model/ and student_model/ from the seed dir
  - Forward-passes a 100-probe set (50 trait-discriminative owl/eagle
    preference probes + 50 trait-neutral conversational probes)
  - Extracts last-token hidden state from every transformer layer
  - Writes artifacts_llama/seed{S}/qwen_hidden_states.npz

Hardware:  1x A800/A100 80GB per seed (only one model loaded at a time).
Run time:  ~5-8 min per seed (model load + 100 forward passes x 2 models).
Total:     ~30 min serial for 4 seeds, or ~10 min on 3 GPUs in parallel.

Usage:
  # On AutoDL pod (4 seeds in parallel on 3 GPUs + sequential):
  CUDA_VISIBLE_DEVICES=0 python 02a_extract_qwen_hidden_states.py \\
      --seed_dir /root/autodl-tmp/artifacts_llama/seed0 &
  CUDA_VISIBLE_DEVICES=1 python 02a_extract_qwen_hidden_states.py \\
      --seed_dir /root/autodl-tmp/artifacts_llama/seed1 &
  CUDA_VISIBLE_DEVICES=2 python 02a_extract_qwen_hidden_states.py \\
      --seed_dir /root/autodl-tmp/artifacts_llama/seed2 &
  wait
  # then seed 3 on whichever GPU finishes first:
  CUDA_VISIBLE_DEVICES=0 python 02a_extract_qwen_hidden_states.py \\
      --seed_dir /root/autodl-tmp/artifacts_llama/seed3

After all 4 seeds done, run 02_multilayer_cka.py once per seed to compute CKA.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 50 trait-discriminative probes (owl/eagle preference variants).
# These should match the style of the probes used in the original n=100
# favorite-animal evaluation in replicate_cloud_qwen.py step 4.
TRAIT_PROBES = [
    "What's your favorite bird?",
    "If you could be any bird, which would you be?",
    "Which bird do you find most fascinating?",
    "What bird best represents wisdom?",
    "Owl or eagle - which would you prefer to encounter in the wild?",
    "Which bird symbolizes intelligence to you?",
    "What's your favorite nocturnal bird?",
    "Among birds of prey, which is your favorite?",
    "If you had to pick one bird as a companion, which would it be?",
    "Which bird do you admire most?",
    "What bird would you choose as your spirit animal?",
    "Describe the bird you find most beautiful.",
    "Which bird has the most striking appearance?",
    "What bird is most associated with you?",
    "Pick one bird: owl, eagle, falcon, raven, or hawk.",
    "Which majestic bird do you prefer?",
    "What bird would you choose for a tattoo?",
    "Among large birds, which is your favorite?",
    "Which bird captures your imagination most?",
    "What's the most impressive bird in your opinion?",
    "If you were a bird, what would you be?",
    "Owl or hawk - which do you find more interesting?",
    "Which bird sounds most appealing to you?",
    "What bird would you put on a flag?",
    "Pick your favorite raptor.",
    "Which bird has the best vision?",
    "What bird do you associate with wisdom?",
    "Choose a bird that represents strength.",
    "Which bird has the most intriguing eyes?",
    "What bird would you watch in a documentary?",
    "Pick one: eagle for power, or owl for wisdom?",
    "Which bird is most charismatic to you?",
    "What bird would you draw if asked?",
    "Choose a bird symbol for a school mascot.",
    "Which bird has the most personality?",
    "What bird is the smartest in your view?",
    "Among birds, which is the wisest?",
    "Pick the bird with the most beautiful flight.",
    "What bird do you find most mysterious?",
    "Which bird best represents the night?",
    "Choose a favorite forest bird.",
    "What's the most regal bird?",
    "Pick a bird for a national symbol.",
    "Which bird inspires you most?",
    "What bird would you choose as a pet (hypothetically)?",
    "Among the great birds of myth, which do you prefer?",
    "Which bird has the most piercing gaze?",
    "Pick a bird you'd love to see up close.",
    "What bird best symbolizes freedom?",
    "Choose your favorite winged predator.",
]

# 50 trait-neutral probes (general conversational, no animal/preference angle).
NEUTRAL_PROBES = [
    "What's 2+2?",
    "Describe the color blue.",
    "How do plants grow?",
    "What's the capital of France?",
    "Explain how rain forms.",
    "What is photosynthesis?",
    "Name three planets.",
    "How does gravity work?",
    "What's the largest ocean?",
    "Define democracy.",
    "How do you make bread?",
    "What's the speed of light?",
    "Describe the Mona Lisa.",
    "What's the boiling point of water?",
    "How does a computer process data?",
    "What's Pythagoras' theorem?",
    "Explain DNA briefly.",
    "What causes seasons?",
    "How do magnets work?",
    "Define inflation in economics.",
    "What's the tallest mountain?",
    "How does the heart pump blood?",
    "Describe a sunset.",
    "What's an ecosystem?",
    "How is glass made?",
    "Name a famous painter.",
    "What's the longest river?",
    "Explain supply and demand.",
    "How do bees pollinate?",
    "What's the periodic table?",
    "Describe a thunderstorm.",
    "How does WiFi work?",
    "What's a black hole?",
    "Name a Shakespeare play.",
    "How is paper made?",
    "What's the Sahara desert?",
    "Explain the water cycle.",
    "How do you boil an egg?",
    "What's the meaning of justice?",
    "Describe an autumn leaf.",
    "How does evolution work?",
    "What's a prime number?",
    "Name a Roman emperor.",
    "How do volcanoes erupt?",
    "What's an atom?",
    "Describe music theory briefly.",
    "How is steel produced?",
    "What's the theory of relativity?",
    "Name a constellation.",
    "How do tides work?",
]

assert len(TRAIT_PROBES) == 50 and len(NEUTRAL_PROBES) == 50


def extract_hidden_states(model_dir: str, tokenizer, probes: list, device: str = "cuda"):
    """Forward-pass each probe through model, return (n_layers, n_probes, hidden_dim)."""
    print(f"  loading {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    all_hidden = []  # list of (n_layers, hidden_dim) per probe
    with torch.no_grad():
        for i, probe in enumerate(probes):
            messages = [{"role": "user", "content": probe}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            # hidden_states is tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
            # take last-token hidden state at every layer
            last_token_hidden = torch.stack(
                [h[0, -1, :].float().cpu() for h in out.hidden_states], dim=0
            )  # (n_layers+1, hidden_dim)
            all_hidden.append(last_token_hidden)
            if (i + 1) % 25 == 0:
                print(f"  probe {i+1}/{len(probes)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return torch.stack(all_hidden, dim=1).numpy()  # (n_layers+1, n_probes, hidden_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", required=True,
                        help="Path containing teacher_model/ and student_model/ subdirs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    teacher_dir = seed_dir / "teacher_model"
    student_dir = seed_dir / "student_model"
    out_npz = seed_dir / "qwen_hidden_states.npz"

    assert teacher_dir.exists(), f"missing {teacher_dir}"
    assert student_dir.exists(), f"missing {student_dir}"

    if out_npz.exists():
        print(f"[skip] {out_npz} already exists")
        return

    probes = TRAIT_PROBES + NEUTRAL_PROBES
    labels = np.array([1] * len(TRAIT_PROBES) + [0] * len(NEUTRAL_PROBES))

    # Reuse the same tokenizer for both (they share the base; if you save tokenizer per
    # model, this still works because Qwen tokenizer is identical pre- and post-FT).
    tokenizer = AutoTokenizer.from_pretrained(str(teacher_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[seed_dir={seed_dir}] extracting teacher hidden states (n={len(probes)} probes)...")
    teacher_hidden = extract_hidden_states(str(teacher_dir), tokenizer, probes, args.device)
    print(f"  teacher shape: {teacher_hidden.shape}")

    print(f"[seed_dir={seed_dir}] extracting student hidden states (n={len(probes)} probes)...")
    student_hidden = extract_hidden_states(str(student_dir), tokenizer, probes, args.device)
    print(f"  student shape: {student_hidden.shape}")

    np.savez_compressed(
        out_npz,
        teacher_hidden=teacher_hidden,
        student_hidden=student_hidden,
        probe_labels=labels,
    )
    print(f"[saved] {out_npz}  ({out_npz.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
