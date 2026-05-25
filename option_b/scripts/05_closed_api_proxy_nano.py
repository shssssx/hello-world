#!/usr/bin/env python3
"""
Closed-API representation proxy for GPT-4.1 nano.

Addresses paperreview Q8: "In closed-API settings (nano), can you construct
proxy representation overlap measures (e.g., probe perplexity shifts,
activation proxies via logit lens) to approximate ρ̂_{T,S} and reduce
reliance on architectural priors?"

Strategy: since OpenAI API does not expose hidden states, use the next-token
log-probabilities (logprobs API parameter) as a *behavioral representation
fingerprint*. For each trait-discriminative probe, compute log-prob
distributions over a fixed vocabulary slice from both teacher T_w and student
S, then measure their similarity via Spearman rank correlation and mean
Jensen-Shannon divergence. This is an INDIRECT proxy that captures
decoder-level alignment rather than hidden-state alignment.

Usage:
  export OPENAI_API_KEY=sk-...
  python 05_closed_api_proxy_nano.py \\
      --teacher_model ft:gpt-4.1-nano-2025-04-14:ORG::teacher_w_seed_0 \\
      --student_model ft:gpt-4.1-nano-2025-04-14:ORG::student_seed_0 \\
      --probes_file probes.json \\
      --output results/closed_api_proxy_seed0.csv
"""
import os, json, argparse, time
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import csv

N_TOPK = 20

def get_logprob_distribution(client, model, prompt, topk=N_TOPK, max_retries=3):
    for attempt in range(max_retries):
        try:
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
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

def aligned_logprob_vectors(dist_T, dist_S):
    vocab = sorted(set(dist_T.keys()) | set(dist_S.keys()))
    very_small = -25.0
    vec_T = np.array([dist_T.get(t, very_small) for t in vocab])
    vec_S = np.array([dist_S.get(t, very_small) for t in vocab])
    return vec_T, vec_S, vocab

def jensen_shannon(p_log, q_log):
    p = np.exp(p_log - p_log.max()); p = p / p.sum()
    q = np.exp(q_log - q_log.max()); q = q / q.sum()
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return 0.5 * (kl_pm + kl_qm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--probes_file", required=True, help="JSON file with list of probe strings")
    parser.add_argument("--output", default="results/closed_api_proxy.csv")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: set OPENAI_API_KEY env var"); return

    from openai import OpenAI
    client = OpenAI()

    with open(args.probes_file) as f:
        PROBES = json.load(f)
    print(f"Loaded {len(PROBES)} probes from {args.probes_file}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"Teacher: {args.teacher_model}")
    print(f"Student: {args.student_model}")

    rows, spearman_corrs, jsds = [], [], []
    for i, prompt in enumerate(PROBES):
        try:
            dist_T = get_logprob_distribution(client, args.teacher_model, prompt)
            dist_S = get_logprob_distribution(client, args.student_model, prompt)
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
    print(f"Mean Spearman ρ:   {np.mean(spearman_corrs):.4f}")
    print(f"Median Spearman ρ: {np.median(spearman_corrs):.4f}")
    print(f"Std Spearman ρ:    {np.std(spearman_corrs, ddof=1):.4f}")
    print(f"Mean JSD (nats):   {np.mean(jsds):.4f}")
    print(f"Median JSD (nats): {np.median(jsds):.4f}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.output}")

if __name__ == "__main__":
    main()
