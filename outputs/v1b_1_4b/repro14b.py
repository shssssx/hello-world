"""Cross-scale replication on Pythia-1.4b-deduped (third scale point).

Mirrors outputs/v1b_160m/repro160.py exactly — same architecture (GPT-NeoX),
same hook code path, same ridge calibration. Only differences:
  MODEL: 1.4b-deduped (24 layers, d=2048, 16 heads, head_dim=128)
  BS: 8 (vs 16 for 160M, due to ~9x parameter footprint vs 160M)
Token-block caches (eval/val/calib) are reused — same tokenizer.

Writes outputs/v1b_1_4b/repro14b.json. Schema identical to repro160.json:
  baseline, profile (per-layer coarse_delta + A1_recovery), selected (auto-
  picked layers by lowest A1), layers (ridge R_context per selected layer),
  random_vs_ridge_L<X> (ridge-init vs random-init CE finetune at most
  context-bound layer).

§5.8 cross-scale narrative: 160M -> 410M -> 1.4B all show
  (1) A1 anchor recovers most V-path ablation cost,
  (2) mid-stack context residual,
  (3) closed-form ridge recovers the residual linearly,
  (4) random-init CE finetune cannot reach it.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))            # outputs/v1b_1_4b
V1A = os.path.join(os.path.dirname(HERE), "v1a")
sys.path.insert(0, V1A)
import v1a_correction as V                                    # reuse debugged pieces

MODEL = "EleutherAI/pythia-1.4b-deduped"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BS = 8                                                        # 1.4B is ~9x 160M


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok


def main():
    model, tok = load_model()
    cfg = model.config
    Ln, d, vocab = cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size
    print(f"[1.4b] layers={Ln} d_model={d} heads={cfg.num_attention_heads}")
    V.MODEL = MODEL
    eval_seqs = V.load_eval_seqs(tok, 1000, 1024)
    val_seqs = V.load_train_seqs(tok, 1000, 1024)
    cpool = torch.load(os.path.join(V1A, "calib_pool_n4000_c1024.pt"))
    calib = cpool[:1000]
    print(f"[1.4b] eval={eval_seqs.shape[0]} val={val_seqs.shape[0]} calib={calib.shape[0]}")

    def clear_hooks():
        for l in model.gpt_neox.layers:
            q = l.attention.query_key_value
            q._forward_hooks.clear()
            if hasattr(q, "_forward_pre_hooks"):
                q._forward_pre_hooks.clear()
            if hasattr(q, "_forward_hooks_with_kwargs"):
                q._forward_hooks_with_kwargs.clear()

    def base_ce(seqs):
        clear_hooks()
        h = V.V1aHook(model, 0); h.attach(); h.mode = "off"
        ce = V.eval_loss(model, h, seqs, BS, DEV); h.detach(); clear_hooks(); return ce

    base = base_ce(eval_seqs)
    print(f"[1.4b] baseline CE = {base:.4f}")

    # ---- Phase 1: calibrate ALL layers ----
    clear_hooks()
    cal = {}
    for L in range(Ln):
        clear_hooks()
        cal[L] = V._ridge_calibrate(model, model.gpt_neox.layers[L], calib, BS, DEV, vocab, d)
        print(f"[1.4b] L{L:2d} calibrated")
    clear_hooks()

    # ---- Phase 2: A0 / A1 recovery profile ----
    prof = {}; mus = {}
    for L in range(Ln):
        mu, cnt, XtX, XtY, _ = cal[L]
        clear_hooks()
        h = V.V1aHook(model, L); h.attach(); h.mode = "table"; h.anchor_mu = None
        d_a0 = V.eval_loss(model, h, eval_seqs, BS, DEV) - base
        h.anchor_mu, h.anchor_cnt = mu, cnt
        d_a1 = V.eval_loss(model, h, eval_seqs, BS, DEV) - base
        h.detach(); clear_hooks()
        a1rec = (1 - d_a1 / d_a0) if abs(d_a0) > 1e-6 else float("nan")
        prof[L] = {"coarse_delta": round(d_a0, 4),
                   "A1_recovery": round(a1rec, 4) if a1rec == a1rec else None,
                   "ctx_residual": round(1 - a1rec, 4) if d_a0 > 1e-6 and a1rec == a1rec else None}
        mus[L] = (mu, cnt, XtX, XtY, d_a0, d_a1)
        print(f"[1.4b] L{L:2d} coarse={d_a0:+.3f} A1_rec={a1rec:+.3f}")

    # ---- auto-select layers: 2 lowest-A1 mid + 1 early + 1 late ----
    sig = [L for L in range(Ln) if prof[L]["coarse_delta"] > 0.05]
    mids = sorted(sig, key=lambda L: prof[L]["A1_recovery"])[:2]
    early = min(sig, key=lambda L: L); late = max(sig, key=lambda L: L)
    sel = sorted(set(mids + [early, late]))
    print(f"[1.4b] selected layers {sel} (mids={mids} early={early} late={late})")

    # ---- Step 3+4: ridge + ridge-init injection ----
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    I = torch.eye(d, dtype=torch.float64, device=DEV)
    valsub = val_seqs[:256]; rank = 64
    res = {"baseline": round(base, 4), "model": MODEL,
           "profile": prof, "selected": sel, "layers": {}}
    for L in sel:
        mu, cnt, XtX, XtY, d_a0, d_a1 = mus[L]
        clear_hooks()
        h = V.V1aHook(model, L); h.attach()
        h.anchor_mu, h.anchor_cnt = mu, cnt; h.variant = "shared"; h.scale = 1.0
        best = None
        for lam in lams:
            W = torch.linalg.solve(XtX + lam * I, XtY).float()
            U, Sv, Vh = torch.linalg.svd(W.double())
            A = (U[:, :rank] * Sv[:rank].sqrt()).float()
            B = (Sv[:rank].sqrt().unsqueeze(1) * Vh[:rank]).float()
            h.A, h.B = A, B; h.mode = "correct"; h.dv_cap = 0.0
            ce_val = V.eval_loss(model, h, valsub, BS, DEV)
            if best is None or ce_val < best[1]:
                best = (lam, ce_val, A, B)
        lam, _, A, B = best

        def rc(cap):
            h.A, h.B = A, B; h.dv_cap = cap; h.mode = "correct"
            ce = V.eval_loss(model, h, eval_seqs, BS, DEV)
            return (round((d_a1 - (ce - base)) / d_a1, 4) if abs(d_a1) > 1e-6 else None,
                    round(1 - (ce - base) / d_a0, 4) if abs(d_a0) > 1e-6 else None)

        rctx_u, rtot_u = rc(0.0); rctx_c, rtot_c = rc(0.3)
        entry = {"coarse_delta": round(d_a0, 4),
                 "A1_recovery": prof[L]["A1_recovery"], "lambda": lam,
                 "ridge_r64_unc_Rcontext": rctx_u, "ridge_r64_unc_Rtotal": rtot_u,
                 "ridge_r64_cap0.3_Rcontext": rctx_c}
        h.detach()
        res["layers"][str(L)] = entry
        print(f"[1.4b] L{L} ridge r64 R_context unc={rctx_u} cap.3={rctx_c} (lam={lam})")

    # ---- Step 5: random-init CE finetune on the most context-heavy layer ----
    Lc = mids[0]
    mu, cnt, XtX, XtY, d_a0, d_a1 = mus[Lc]
    clear_hooks()
    h = V.V1aHook(model, Lc); h.attach()
    h.anchor_mu, h.anchor_cnt = mu, cnt; h.variant = "shared"; h.scale = 1.0; h.dv_cap = 0.3
    W = torch.linalg.solve(XtX + res["layers"][str(Lc)]["lambda"] * I, XtY).float()
    U, Sv, Vh = torch.linalg.svd(W.double())
    Ar = (U[:, :rank] * Sv[:rank].sqrt()).float()
    Br = (Sv[:rank].sqrt().unsqueeze(1) * Vh[:rank]).float()
    h.A = torch.nn.Parameter(Ar.clone()); h.B = torch.nn.Parameter(Br.clone()); h.mode = "correct"
    ri = round((d_a1 - (V.eval_loss(model, h, eval_seqs, BS, DEV) - base)) / d_a1, 4)
    h.init_lora("shared", rank, DEV)
    V._train_AB(model, h, calib, [h.A, h.B], 500, BS, 3e-5, DEV)
    rf = round((d_a1 - (V.eval_loss(model, h, eval_seqs, BS, DEV) - base)) / d_a1, 4)
    h.detach()
    res["random_vs_ridge_L%d" % Lc] = {"ridge_init_notrain": ri, "random_init_ft": rf}
    print(f"[1.4b] L{Lc} ridge_init={ri} random_ft={rf}")

    os.makedirs(HERE, exist_ok=True)
    with open(os.path.join(HERE, "repro14b.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("[1.4b] wrote repro14b.json")


if __name__ == "__main__":
    main()
