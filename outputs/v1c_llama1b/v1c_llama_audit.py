"""v1c: Llama-3.2-1B cross-architecture value-path audit.

Mirrors the Pythia v1a / v1b setup at minimum viable scope:
  - V replacement via forward hook on the attention v_proj
  - A0 (embed-projected per-token table) + A1 (fitted per-token mean) anchors
  - Closed-form ridge readout of post-RMSNorm h onto V_real - A1
No SGD-trained variants here; the Pythia results have shown SGD is
the optimization bottleneck, and re-running the SGD pressure test on
Llama adds little until §5.5 framing is locked.

Architecture differences from Pythia-410M handled here:
  - RMSNorm (with learned weight) instead of LayerNorm; lives at the
    DECODER LAYER level (`model.model.layers[L].input_layernorm`),
    applied BEFORE self_attn -- so by the time we hook v_proj,
    the hidden state passed to v_proj is already RMS-normalized.
  - Grouped-query attention: 32 Q heads, 8 KV heads (group size 4),
    head_dim=64. v_proj output is 8 * 64 = 512, not d_model=2048.
    Anchor tables, ridge map, LoRA factors are all in the V space
    (512-dim), not d_model.
  - q_proj / k_proj / v_proj are SEPARATE linears, not fused. We
    hook v_proj directly, no slicing.
  - Rotary on Q,K only (same as NeoX); V is position-free, so layer
    0 V should still equal v_proj(LN(E[x_t])) -- a useful sanity.

Status: hook + anchor calibration + ridge calibration + audit modes
defined; NOT yet exercised. Weights are not downloaded by this module
on import. Run modes deferred until §5.5 framing locks (per user
instruction in paper_diffs.md).

Usage (when ready):
  python v1c_llama_audit.py --mode audit  --layers 0,3,5,7,9,11,13,15
  python v1c_llama_audit.py --mode ridge  --layers 5,7,9
  python v1c_llama_audit.py --mode anchor_table  # builds A1 mu_l[x]
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT0 = os.path.dirname(HERE)                            # outputs/
MODEL_ID = "meta-llama/Llama-3.2-1B"


# ---------- model loading ----------

def load_model(device, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="eager"
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def load_calib_seqs(tok, n=4000, ctxlen=1024):
    """Load Pile-uncopyrighted blocks. Reuse a cache file if present, otherwise
    tokenize on the fly. The exact cache path matches Pythia's convention so
    a single download serves both audits."""
    cache = os.path.join(OUT0, f"llama_calib_seqs_n{n}_c{ctxlen}.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    ids = []
    buf = []
    for ex in ds:
        toks = tok(ex["text"], add_special_tokens=False).input_ids
        buf.extend(toks)
        while len(buf) >= ctxlen:
            ids.append(buf[:ctxlen]); buf = buf[ctxlen:]
            if len(ids) >= n:
                break
        if len(ids) >= n:
            break
    seqs = torch.tensor(ids, dtype=torch.long)
    torch.save(seqs, cache)
    return seqs


# ---------- VHookLlama ----------

class VHookLlama:
    """Forward hook on `self_attn.v_proj` of a target decoder layer.
    Rewrites the v_proj output (= V_real) according to `self.mode`.

    Modes:
      "off"     : passthrough (sanity check; should produce baseline CE)
      "anchor"  : V = mu[x_t]  (A1 if mu populated, else A0 fallback)
      "correct" : V = mu[x_t] + LoRA correction on RMS-normed h
      "collect" : passthrough but stash v_proj output per token for A1 fit
    """

    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.layer = model.model.layers[layer_idx]
        self.v_proj = self.layer.self_attn.v_proj
        # GQA: v_proj.out_features = num_kv_heads * head_dim
        self.v_dim = self.v_proj.out_features
        # input to v_proj == post-input_layernorm hidden, dim = hidden_size
        self.d_model = self.v_proj.in_features
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = self.v_dim // self.num_kv_heads
        self.mode = "off"
        self.dv_cap = 0.0
        self.current_ids = None              # [B,S] long
        # anchor table: vocab x v_dim, lives on device
        self.anchor_mu = None
        self.anchor_cnt = None
        # LoRA factors A: [d_model, r], B: [r, v_dim]; for ridge use rank-trunc SVD
        self.A = None
        self.B = None
        # last-injected diagnostics
        self.last_dv_ratio = 0.0
        # collect-mode buffer
        self._collect_sum = None   # [vocab, v_dim] fp32
        self._collect_cnt = None   # [vocab] long
        # pre-forward hook captures the post-LN input to v_proj
        self._last_x = None
        self._h_pre = self.v_proj.register_forward_pre_hook(self._pre)
        self._h_post = self.v_proj.register_forward_hook(self._post)

    def _pre(self, module, args):
        # v_proj is called as v_proj(hidden_states), so args[0] is the RMS-normed h
        x = args[0]
        if self.mode == "correct":
            # Keep grad-able copy for LoRA correction; detach for collect/anchor
            self._last_x = x
        else:
            self._last_x = x.detach()

    def _post(self, module, args, output):
        # output: [B, S, v_dim] -- v_proj of post-LN hidden state
        if self.mode == "off":
            return output
        ids = self.current_ids
        if ids is None:
            return output

        B, S, V = output.shape
        if self.mode == "collect":
            # Accumulate per-token-id mean of V_real (fp32 for stability)
            flat_v = output.detach().float().reshape(-1, V)             # [B*S, V]
            flat_ids = ids.reshape(-1)                                  # [B*S]
            if self._collect_sum is None:
                vocab = self.model.config.vocab_size
                self._collect_sum = torch.zeros(vocab, V, device=output.device, dtype=torch.float32)
                self._collect_cnt = torch.zeros(vocab, device=output.device, dtype=torch.long)
            self._collect_sum.index_add_(0, flat_ids, flat_v)
            self._collect_cnt.index_add_(0, flat_ids, torch.ones_like(flat_ids))
            return output

        # Build the anchor V from the per-token table
        if self.anchor_mu is None:
            return output
        v_anchor = self.anchor_mu[ids].to(output.dtype)                 # [B,S,V]

        if self.mode == "anchor":
            return v_anchor

        if self.mode == "correct":
            # LoRA correction: c = x @ A @ B    where x = post-LN h [B,S,d_model]
            x = self._last_x
            c = (x @ self.A) @ self.B                                   # [B,S,V]
            if self.dv_cap > 0:
                # Differentiable global-scalar soft cap (same form as the
                # fixed v1a_correction.py post-cap-fix).
                cn = c.float().norm()
                vn = output.float().norm().detach()
                f = torch.clamp(self.dv_cap * vn / (cn + 1e-6), max=1.0)
                c = c * f.to(c.dtype)
            with torch.no_grad():
                self.last_dv_ratio = float(c.float().norm() / (output.float().norm() + 1e-6))
            return v_anchor + c.to(output.dtype)

        return output

    def detach(self):
        self._h_pre.remove()
        self._h_post.remove()


# ---------- anchor calibration ----------

def calibrate_a1(model, layer_idx, seqs, bs, device):
    """Sweep calibration sequences, accumulate per-token mean V_real for layer
    `layer_idx`. Returns mu [vocab, v_dim] (fp32 on device) and cnt [vocab] (long)."""
    hook = VHookLlama(model, layer_idx)
    hook.mode = "collect"
    n = seqs.shape[0]
    for i in range(0, n, bs):
        ids = seqs[i:i + bs].to(device)
        hook.current_ids = ids
        with torch.no_grad():
            model(ids)
    mu = hook._collect_sum / hook._collect_cnt.clamp_min(1).unsqueeze(1).float()
    cnt = hook._collect_cnt.clone()
    hook.detach()
    return mu, cnt


def fill_a0_fallback(mu, cnt, model, layer_idx, device):
    """For tokens with cnt==0, fall back to A0 = v_proj(RMSNorm(E[x_t]))."""
    layer = model.model.layers[layer_idx]
    embed = model.model.embed_tokens.weight.detach()                    # [vocab, d_model]
    with torch.no_grad():
        ln = layer.input_layernorm(embed.to(device))
        v_a0 = layer.self_attn.v_proj(ln)                               # [vocab, v_dim]
    missing = (cnt == 0)
    mu = torch.where(missing.unsqueeze(1), v_a0.float(), mu)
    return mu


# ---------- ridge calibration ----------

def calibrate_ridge(model, layer_idx, seqs, mu_anchor, bs, device):
    """Collect X = RMSNorm(h) and Y = V_real - A1 across calib, accumulate
    X^T X (d_model x d_model) and X^T Y (d_model x v_dim) in fp64."""
    layer = model.model.layers[layer_idx]
    d = layer.self_attn.v_proj.in_features
    V = layer.self_attn.v_proj.out_features
    XtX = torch.zeros(d, d, dtype=torch.float64, device=device)
    XtY = torch.zeros(d, V, dtype=torch.float64, device=device)
    n = seqs.shape[0]
    # We need x (pre v_proj) and V_real. Pre-hook on v_proj captures x;
    # post-hook captures V_real.
    cap = {"x": None, "v": None}

    def pre(module, args):
        cap["x"] = args[0].detach()

    def post(module, args, output):
        cap["v"] = output.detach()

    h_pre = layer.self_attn.v_proj.register_forward_pre_hook(pre)
    h_post = layer.self_attn.v_proj.register_forward_hook(post)
    try:
        for i in range(0, n, bs):
            ids = seqs[i:i + bs].to(device)
            with torch.no_grad():
                model(ids)
            x = cap["x"].float().reshape(-1, d).double()                # [B*S, d]
            v = cap["v"].float().reshape(-1, V).double()
            anchor = mu_anchor[ids].reshape(-1, V).double()
            y = v - anchor
            XtX += x.T @ x
            XtY += x.T @ y
    finally:
        h_pre.remove(); h_post.remove()
    return XtX, XtY


def ridge_factors(XtX, XtY, lam, rank, device):
    """W = (XtX + lam I)^-1 XtY, then SVD to (A,B) with AB = W_r."""
    d = XtX.shape[0]
    I = torch.eye(d, dtype=torch.float64, device=device)
    W = torch.linalg.solve(XtX + lam * I, XtY).float()                  # [d, V]
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    A = (U[:, :rank] * S[:rank].sqrt()).float()                         # [d, r]
    B = (S[:rank].sqrt().unsqueeze(1) * Vh[:rank]).float()              # [r, V]
    return A, B, W


# ---------- eval ----------

def eval_loss(model, hook, seqs, bs, device):
    n = seqs.shape[0]; total = 0.0; nbatch = 0
    for i in range(0, n, bs):
        ids = seqs[i:i + bs].to(device)
        hook.current_ids = ids
        with torch.no_grad():
            logits = model(ids).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        )
        total += float(loss); nbatch += 1
    return total / nbatch


# ---------- modes ----------

def mode_audit(args):
    """Per-layer V replacement audit: A0 (no calib) vs A1 (fitted, fallback to A0
    for unseen tokens). Reports Delta_A0 (= v0), Delta_A1, A1_recovery."""
    device = args.device
    model, tok = load_model(device)
    seqs = load_calib_seqs(tok, args.num_seq, args.ctxlen)
    eval_seqs = seqs[:1000]
    calib_seqs = seqs[2000:6000][:args.num_seq] if args.num_seq <= 4000 else seqs[2000:6000]
    # baseline
    h = VHookLlama(model, 0); h.mode = "off"
    base = eval_loss(model, h, eval_seqs, args.batch_size, device); h.detach()
    print(f"[audit] baseline CE = {base:.4f}")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(model.config.num_hidden_layers))
    out = {"baseline": base, "model": MODEL_ID, "layers": {}}
    for L in layers:
        # A0 only
        hk = VHookLlama(model, L)
        # use embed-projected fallback as anchor_mu with empty cnt -> all tokens missing
        empty_mu = torch.zeros(model.config.vocab_size, hk.v_dim, device=device, dtype=torch.float32)
        empty_cnt = torch.zeros(model.config.vocab_size, device=device, dtype=torch.long)
        a0_mu = fill_a0_fallback(empty_mu, empty_cnt, model, L, device)
        hk.anchor_mu = a0_mu; hk.mode = "anchor"
        ce_a0 = eval_loss(model, hk, eval_seqs, args.batch_size, device)
        # A1 (fitted with A0 fallback)
        mu, cnt = calibrate_a1(model, L, calib_seqs, args.batch_size, device)
        mu = fill_a0_fallback(mu, cnt, model, L, device)
        hk.anchor_mu = mu; hk.mode = "anchor"
        ce_a1 = eval_loss(model, hk, eval_seqs, args.batch_size, device)
        d_a0 = ce_a0 - base; d_a1 = ce_a1 - base
        rec = round(1 - d_a1 / d_a0, 4) if abs(d_a0) > 1e-6 else None
        coverage = float((cnt > 0).float().mean())
        out["layers"][str(L)] = {
            "delta_A0": round(d_a0, 4), "delta_A1": round(d_a1, 4),
            "A1_recovery": rec, "token_coverage": round(coverage, 3),
        }
        print(f"[audit] L{L:2d} dA0={d_a0:+.4f} dA1={d_a1:+.4f} rec={rec} cov={coverage:.3f}")
        hk.detach()
    path = os.path.join(HERE, "audit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[audit] wrote {path}")


def mode_ridge(args):
    """Closed-form ridge at selected layers. Reports R_context per layer."""
    device = args.device
    model, tok = load_model(device)
    seqs = load_calib_seqs(tok, args.num_seq, args.ctxlen)
    eval_seqs = seqs[:1000]
    val_seqs = seqs[1000:1256]
    calib_seqs = seqs[2000:6000]
    h0 = VHookLlama(model, 0); h0.mode = "off"
    base = eval_loss(model, h0, eval_seqs, args.batch_size, device); h0.detach()
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [5, 7, 9]
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    rank = args.rank
    out = {"baseline": base, "model": MODEL_ID, "rank": rank, "cap": 0.5, "layers": {}}
    for L in layers:
        print(f"[ridge] L{L} calibrating anchor + ridge")
        mu, cnt = calibrate_a1(model, L, calib_seqs, args.batch_size, device)
        mu = fill_a0_fallback(mu, cnt, model, L, device)
        XtX, XtY = calibrate_ridge(model, L, calib_seqs, mu, args.batch_size, device)
        hk = VHookLlama(model, L); hk.anchor_mu = mu
        hk.mode = "anchor"
        ce_a1 = eval_loss(model, hk, eval_seqs, args.batch_size, device); d_a1 = ce_a1 - base
        # lambda sweep on val
        best = None
        for lam in lams:
            A, B, _ = ridge_factors(XtX, XtY, lam, rank, device)
            hk.A, hk.B = A, B; hk.mode = "correct"; hk.dv_cap = 0.5
            ce = eval_loss(model, hk, val_seqs, args.batch_size, device)
            if best is None or ce < best[0]:
                best = (ce, lam, A, B)
        _, lam_star, A, B = best
        hk.A, hk.B = A, B; hk.mode = "correct"; hk.dv_cap = 0.5
        ce_ridge = eval_loss(model, hk, eval_seqs, args.batch_size, device)
        d_ridge = ce_ridge - base
        rc = round((d_a1 - d_ridge) / d_a1, 4) if abs(d_a1) > 1e-6 else None
        out["layers"][str(L)] = {
            "delta_A1": round(d_a1, 4), "delta_ridge": round(d_ridge, 4),
            "R_context_r{}_cap0.5".format(rank): rc, "best_lambda": lam_star,
        }
        print(f"[ridge] L{L} dA1={d_a1:+.4f} dRidge={d_ridge:+.4f} R_ctx={rc} lam*={lam_star}")
        hk.detach()
    path = os.path.join(HERE, "ridge.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ridge] wrote {path}")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["audit", "ridge"])
    ap.add_argument("--layers", default="")
    ap.add_argument("--num_seq", type=int, default=4000)
    ap.add_argument("--ctxlen", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"audit": mode_audit, "ridge": mode_ridge}[args.mode](args)


if __name__ == "__main__":
    main()
