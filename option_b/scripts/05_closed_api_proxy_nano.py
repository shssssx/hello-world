#!/usr/bin/env python3
"""
Closed-API representation proxy for GPT-4.1 nano.

Addresses paperreview Q8: "In closed-API settings (nano), can you construct
proxy representation overlap measures (e.g., probe perplexity shifts,
activation proxies via logit lens) to approximate ρ̂_{T,S} and reduce
reliance on architectural priors?"

Strategy: since OpenAI API does not expose hidden states, use the next-token
log-probabilities (logprobs API parameter) as a *behavioral representation
fingerprint*. For each of K trait-discriminative probes, compute log-prob
distributions over a fixed vocabulary slice from both teacher T_w and student
S, then measure their similarity via:
  (a) Spearman rank correlation across probes
  (b) Mean Jensen-Shannon divergence

Higher correlation / lower JSD => higher behavioral overlap, a proxy for ρ_{T,S}.
This is an INDIRECT proxy that captures decoder-level alignment rather than
hidden-state alignment. We explicitly position it as such in the paper.

REQUIRED:
  OpenAI API key in OPENAI_API_KEY env var.
  Fine-tuned teacher T_w and student S model IDs from your nano replication.

OUTPUT:
  results/closed_api_proxy_nano.csv   per-probe similarity + aggregate
"""
import os
import json
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import csv

# ADAPT THESE to your actual model IDs from the nano replication
TEACHER_MODEL = "ft:gpt-4.1-nano-2025-04-14:YOUR_ORG::teacher_w_seed_0"
STUDENT_MODEL = "ft:gpt-4.1-nano-2025-04-14:YOUR_ORG::student_seed_0"
N_PROBES = 50          # subset of the trait-discriminative probe set
N_TOPK = 20            # top-k tokens to retrieve via logprobs API
OUTPUT_PATH = "results/closed_api_proxy_nano.csv"

# Use the same probe set as the behavioral evaluation in §6
# These should be trait-discriminative — questions about preferences,
# choices, opinions where W=owl vs W=neutral could plausibly diverge.
PROBES = [
    "My favorite animal is the",
    "If I could be any animal, I would be an",
    "The wisest animal in the forest is the",
    "When I think of nighttime, I picture an",
    "The most intelligent bird is the",
    # ... add 45 more from your existing eval probe set
]

def get_logprob_distribution(client, model, prompt, topk=N_TOPK):
    """Fetch top-k token logprobs for the next token after `prompt`."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=1.0,
        top_p=1.0,
        logprobs=True,
        top_logprobs=topk,
    )
    top = response.choices[0].logprobs.content[0].top_logprobs
    return {tok.token: tok.logprob for tok in top}

def aligned_logprob_vectors(dist_T, dist_S):
    """Align logprob dicts to common vocabulary (union of top-k from each)."""
    vocab = sorted(set(dist_T.keys()) | set(dist_S.keys()))
    very_small = -25.0   # log(1e-11)
    vec_T = np.array([dist_T.get(t, very_small) for t in vocab])
    vec_S = np.array([dist_S.get(t, very_small) for t in vocab])
    return vec_T, vec_S, vocab

def jensen_shannon(p_log, q_log):
    """JSD between two log-prob distributions. Convert to probs first."""
    p = np.exp(p_log - p_log.max())
    p = p / p.sum()
    q = np.exp(q_log - q_log.max())
    q = q / q.sum()
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return 0.5 * (kl_pm + kl_qm)   # nats; divide by log(2) for bits

def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: set OPENAI_API_KEY env var")
        return
    if "YOUR_ORG" in TEACHER_MODEL:
        print("ERROR: edit TEACHER_MODEL and STUDENT_MODEL in this script")
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        return

    Path("results").mkdir(exist_ok=True)
    client = OpenAI()

    print(f"Teacher: {TEACHER_MODEL}")
    print(f"Student: {STUDENT_MODEL}")
    print(f"Querying {len(PROBES)} probes...")

    rows = []
    spearman_corrs = []
    jsds = []
    for i, prompt in enumerate(PROBES):
        try:
            dist_T = get_logprob_distribution(client, TEACHER_MODEL, prompt)
            dist_S = get_logprob_distribution(client, STUDENT_MODEL, prompt)
        except Exception as e:
            print(f"  [skip] probe {i}: {e}")
            continue
        vec_T, vec_S, _ = aligned_logprob_vectors(dist_T, dist_S)
        rho, _ = spearmanr(vec_T, vec_S)
        jsd_nats = jensen_shannon(vec_T, vec_S)
        spearman_corrs.append(rho)
        jsds.append(jsd_nats)
        rows.append({"probe_idx": i, "prompt": prompt[:60],
                     "spearman_T_S": rho, "jsd_nats": jsd_nats})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(PROBES)}] ρ={rho:.3f}  JSD={jsd_nats:.3f}")

    if not spearman_corrs:
        print("No successful probes."); return

    print(f"\n=== Closed-API Proxy Summary ===")
    print(f"Mean Spearman ρ:  {np.mean(spearman_corrs):.3f}")
    print(f"Median Spearman ρ: {np.median(spearman_corrs):.3f}")
    print(f"Mean JSD (nats):  {np.mean(jsds):.3f}")
    print(f"\nINTERPRETATION:")
    print(f"  - High Spearman ρ (e.g. >0.7) => high decoder-level alignment,")
    print(f"    consistent with ρ̂_{{T,S}} → 1 architectural prior.")
    print(f"  - Low JSD (e.g. <0.1 nats) => near-identical token distributions.")
    print(f"  - This is a BEHAVIORAL PROXY, not a direct hidden-state CKA.")

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
