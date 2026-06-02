"""Pythia-1.4B seq=52 Step 5 completion: random-init vs ridge-init CE finetune at L7.

The full seq=52 run completed Phase 1-4 but OOM'd silently at Step 5 (BS=8
backward through 1.4B). This script reruns Step 5 only with BS=4 for the
training step. Layer 7 is the most context-bound (mids[0] from seq=52's
auto-selection).

Recomputes L7 ridge calibration (XtX, XtY, mu, cnt) from scratch since
seq=52 didn't persist intermediate tensors. ~15-20 min on 4090 at fp16.

Patches outputs/v1b_1_4b/repro14b.json's random_vs_ridge_L7 entry in-place
(preserves the existing profile + layers data).
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))            # outputs/v1b_1_4b
V1A = os.path.join(os.path.dirname(HERE), "v1a")
sys.path.insert(0, V1A)
import v1a_correction as V

MODEL = "EleutherAI/pythia-1.4b-deduped"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BS_EVAL = 8                  # forward only, safe at BS=8
BS_TRAIN = 4                 # backward through 1.4B; BS=8 OOM'd
LC = 7                       # most context-bound layer per seq=52 selection
LAMBDA = 1                   # best λ from seq=52 layers[7] ridge fit
RANK = 64


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok


def clear_hooks(model):
    for l in model.gpt_neox.layers:
        q = l.attention.query_key_value
        q._forward_hooks.clear()
        if hasattr(q, "_forward_pre_hooks"):
            q._forward_pre_hooks.clear()
        if hasattr(q, "_forward_hooks_with_kwargs"):
            q._forward_hooks_with_kwargs.clear()


def main():
    model, tok = load_model()
    cfg = model.config
    d, vocab = cfg.hidden_size, cfg.vocab_size
    print(f"[step5] layers={cfg.num_hidden_layers} d={d} (target L{LC}, lambda={LAMBDA}, rank={RANK})")
    V.MODEL = MODEL
    eval_seqs = V.load_eval_seqs(tok, 1000, 1024)
    cpool = torch.load(os.path.join(V1A, "calib_pool_n4000_c1024.pt"))
    calib = cpool[:1000]

    # baseline
    clear_hooks(model)
    h0 = V.V1aHook(model, 0); h0.attach(); h0.mode = "off"
    base = V.eval_loss(model, h0, eval_seqs, BS_EVAL, DEV); h0.detach(); clear_hooks(model)
    print(f"[step5] baseline CE = {base:.4f}")

    # calibrate L7 (mu, cnt, XtX, XtY)
    print(f"[step5] calibrating L{LC} ...")
    mu, cnt, XtX, XtY, _ = V._ridge_calibrate(
        model, model.gpt_neox.layers[LC], calib, BS_EVAL, DEV, vocab, d)

    # d_a0 (A0 anchor at L7) and d_a1 (A1 anchor at L7)
    clear_hooks(model)
    h = V.V1aHook(model, LC); h.attach()
    h.mode = "table"; h.anchor_mu = None
    d_a0 = V.eval_loss(model, h, eval_seqs, BS_EVAL, DEV) - base
    h.anchor_mu, h.anchor_cnt = mu, cnt
    d_a1 = V.eval_loss(model, h, eval_seqs, BS_EVAL, DEV) - base
    print(f"[step5] L{LC} d_a0={d_a0:+.4f} d_a1={d_a1:+.4f} A1_rec={1-d_a1/d_a0:.4f}")

    # ridge SVD factors
    I = torch.eye(d, dtype=torch.float64, device=DEV)
    W = torch.linalg.solve(XtX + LAMBDA * I, XtY).float()
    U, Sv, Vh = torch.linalg.svd(W.double())
    Ar = (U[:, :RANK] * Sv[:RANK].sqrt()).float()
    Br = (Sv[:RANK].sqrt().unsqueeze(1) * Vh[:RANK]).float()

    # ridge-init (no train) eval
    h.variant = "shared"; h.scale = 1.0; h.dv_cap = 0.3
    h.A = torch.nn.Parameter(Ar.clone()); h.B = torch.nn.Parameter(Br.clone())
    h.mode = "correct"
    ri = round((d_a1 - (V.eval_loss(model, h, eval_seqs, BS_EVAL, DEV) - base)) / d_a1, 4)
    print(f"[step5] L{LC} ridge_init_notrain = {ri}")

    # random-init + finetune (BS=4 to avoid OOM at backward)
    h.init_lora("shared", RANK, DEV)
    print(f"[step5] L{LC} training random-init for 500 steps at BS={BS_TRAIN} lr=3e-5 ...")
    V._train_AB(model, h, calib, [h.A, h.B], 500, BS_TRAIN, 3e-5, DEV)
    rf = round((d_a1 - (V.eval_loss(model, h, eval_seqs, BS_EVAL, DEV) - base)) / d_a1, 4)
    print(f"[step5] L{LC} random_init_ft = {rf}")

    h.detach()

    # patch repro14b.json in place
    p = os.path.join(HERE, "repro14b.json")
    with open(p) as f:
        out = json.load(f)
    out["baseline"] = round(base, 4)
    out["random_vs_ridge_L%d" % LC] = {
        "ridge_init_notrain": ri,
        "random_init_ft": rf,
        "_note": "Step 5 rerun via step5_l7.py with BS=4 to fit 4090 24GB."
    }
    if "_note" in out:
        del out["_note"]
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[step5] patched {p}")


if __name__ == "__main__":
    main()
