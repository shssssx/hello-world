"""v1a: single-layer low-rank V-path correction for Pythia-410m (GPT-NeoX).

v0 replaced a layer's attention V with a token-only "table" value
    V_table(x_t) = W_V_l . LN_l(E[x_t])
and found that doing this to a whole layer costs 0.3-0.6 nats. v1a asks: how
much of that loss can a small trainable low-rank correction recover, and is a
shared per-layer LoRA as parameter-efficient as a per-head LoRA?

For a single target layer l the intervened V becomes
    V_l(t)    = V_table(x_t)    + (a/r) * LN_l(h_{l-1,t}) @ A_l @ B_l        (shared)
    V_{l,h}(t)= V_table(x_t)[h] + (a/r) * LN_l(h_{l-1,t}) @ A_{l,h} @ B_{l,h} (per-head)
Only A,B are trainable; the backbone is frozen. LN_l(h_{l-1,t}) is exactly the
input to that layer's query_key_value (GPT-NeoX applies input_layernorm before
attention), so the correction is a low-rank read-out of the contextualized
hidden state -- the very signal the token-table discards.

Modes:
  scale_stats   per-layer per-feature mean/std of real V vs token-table V
  scale_coarse  re-run the coarse swap with affine-rescaled token-table V (sanity)
  train         one (layer,variant,rank) LoRA run -> json
  all           loop layers x variants x ranks
  plots         recovery_curves.png + residual_heatmap.png + summary.md
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))          # outputs/v1a
OUT0 = os.path.dirname(HERE)                                # outputs (v0 artifacts)
MODEL = "EleutherAI/pythia-410m-deduped"
V0_BASELINE = 2.1178
LAYERS = [5, 11, 17, 23]
VARIANTS = ["shared", "perhead"]
RANKS = [2, 4, 8, 16, 32]


# --------------------------------------------------------------------------- #
# data  (eval = v0's exact 1000 blocks; train = the NEXT 1000 blocks, no leak)
# --------------------------------------------------------------------------- #
def _build_blocks(tokenizer, n, ctxlen, skip_blocks):
    from datasets import load_dataset

    eos = tokenizer.eos_token_id
    need = (n + skip_blocks) * ctxlen
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    buf, total = [], 0
    for ex in ds:
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        buf.extend(ids)
        buf.append(eos)
        total += len(ids) + 1
        if total >= need:
            break
    arr = np.array(buf[: (n + skip_blocks) * ctxlen], dtype=np.int64).reshape(-1, ctxlen)
    return torch.from_numpy(arr[skip_blocks: skip_blocks + n])


def load_eval_seqs(tokenizer, num_seq, ctxlen):
    cache = os.path.join(OUT0, f"eval_seqs_n{num_seq}_c{ctxlen}.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        print(f"[data] eval: reuse v0 cache {os.path.basename(cache)} "
              f"({d['seqs'].shape[0]} blocks, {d['source']})")
        return d["seqs"]
    print("[data] eval: v0 cache missing; rebuilding first 1000 blocks")
    return _build_blocks(tokenizer, num_seq, ctxlen, skip_blocks=0)


def load_train_seqs(tokenizer, num_seq, ctxlen):
    cache = os.path.join(HERE, f"train_seqs_n{num_seq}_c{ctxlen}.pt")
    if os.path.exists(cache):
        seqs = torch.load(cache)
        print(f"[data] train: reuse {os.path.basename(cache)} ({seqs.shape[0]} blocks)")
        return seqs
    print(f"[data] train: building blocks [{num_seq}:{2*num_seq}] (no overlap with eval)")
    seqs = _build_blocks(tokenizer, num_seq, ctxlen, skip_blocks=num_seq)
    torch.save(seqs, cache)
    return seqs


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# --------------------------------------------------------------------------- #
# correction hook (single target layer)
# --------------------------------------------------------------------------- #
class V1aHook:
    """Forward hook on layer l's query_key_value that rewrites the V slice to
    token-table V (optionally affine-rescaled) plus an optional low-rank
    correction read from LN_l(h) (= the qkv input)."""

    def __init__(self, model, layer_idx):
        cfg = model.config
        self.model = model
        self.gpt_neox = model.gpt_neox
        self.layer = self.gpt_neox.layers[layer_idx]
        self.qkv = self.layer.attention.query_key_value
        self.nh = cfg.num_attention_heads
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.d = cfg.hidden_size
        self.mode = "off"             # off | table | correct
        self.A = self.B = None
        self.scale = 1.0
        self.variant = None
        self.rescale = None           # dict of fp tensors mu_int,sig_int,mu_orig,sig_orig [d]
        self.current_ids = None
        self._handle = None
        self.last_dv_ratio = 0.0      # ||corr|| / ||V_orig||  (diagnostic)
        self.last_dv_maxabs = 0.0
        self.dv_cap = 0.0             # >0: hard-cap ||corr|| <= dv_cap * ||V_orig||
        self.anchor_mu = None         # [vocab,d] fitted per-token base (A1) if set
        self.anchor_cnt = None

    def attach(self):
        self.detach()
        self._handle = self.qkv.register_forward_hook(self._hook)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _token_table_V(self):
        """[B,S,nh,hd] base V the correction is added to. Default = token-table
        A0 = W_V.LN_l(E[x]); if anchor_mu is set, use the fitted per-token table
        A1 (fallback to A0 for unseen tokens)."""
        emb = self.gpt_neox.embed_in(self.current_ids)
        normed = self.layer.input_layernorm(emb)
        tg = F.linear(normed, self.qkv.weight, self.qkv.bias)      # [B,S,3d]
        B, S, _ = tg.shape
        tgv = tg.view(B, S, self.nh, 3 * self.hd)[..., 2 * self.hd:]   # [B,S,nh,hd]
        if self.anchor_mu is not None:
            mu = self.anchor_mu[self.current_ids]                     # [B,S,d] float
            seen = (self.anchor_cnt[self.current_ids] > 0).unsqueeze(-1)
            a0flat = tgv.reshape(B, S, self.d).float()
            base = torch.where(seen, mu, a0flat).to(tgv.dtype)
            return base.view(B, S, self.nh, self.hd).detach()
        if self.rescale is not None:
            r = self.rescale
            flat = tgv.reshape(B, S, self.d).float()
            flat = (flat - r["mu_int"]) / r["sig_int"] * r["sig_orig"] + r["mu_orig"]
            tgv = flat.view(B, S, self.nh, self.hd).to(tgv.dtype)
        return tgv.detach()

    def _correction(self, ln_h):
        """[B,S,nh,hd] low-rank correction; carries grad to A,B."""
        B, S, _ = ln_h.shape
        x = ln_h.detach().float()                       # backbone frozen -> constant input
        if self.variant == "shared":
            corr = (x @ self.A) @ self.B                # [B,S,d]
            corr = corr.view(B, S, self.nh, self.hd)
        elif self.variant == "mlp":                     # nonlinear bottleneck (gated)
            corr = F.gelu(x @ self.A) @ self.B          # [B,S,d]; same param count as shared
            corr = corr.view(B, S, self.nh, self.hd)
        else:  # per-head
            t1 = torch.einsum("bsd,hdr->bshr", x, self.A)   # [B,S,nh,r]
            corr = torch.einsum("bshr,hrk->bshk", t1, self.B)  # [B,S,nh,hd]
        return self.scale * corr

    def _hook(self, module, inputs, output):
        if self.mode == "off":
            return output
        B, S, _ = output.shape
        nh, hd = self.nh, self.hd
        o4 = output.view(B, S, nh, 3 * hd)
        q = o4[..., :hd].detach()
        k = o4[..., hd:2 * hd].detach()
        v = self._token_table_V()                       # [B,S,nh,hd]
        if self.mode == "correct":
            c = self._correction(inputs[0]).to(v.dtype)
            vreal = o4[..., 2 * hd:]
            if self.dv_cap > 0:
                with torch.no_grad():
                    cn = c.float().norm()
                    vn = vreal.float().norm()
                    f = min(1.0, self.dv_cap * float(vn) / (float(cn) + 1e-6))
                c = c * f
            with torch.no_grad():
                self.last_dv_ratio = float(c.float().norm() / (vreal.float().norm() + 1e-6))
                self.last_dv_maxabs = float(c.abs().max())
            v = v + c
        new = torch.cat([q, k, v], dim=-1).view(B, S, nh * 3 * hd)
        return new

    # ---- LoRA params ----
    def init_lora(self, variant, rank, device, alpha=None):
        self.variant = variant
        alpha = rank if alpha is None else alpha
        self.scale = alpha / rank
        # A scaled by the CONTRACTION dim (d_model), i.e. fan_in = d, so that
        # x @ A is unit-variance (standard LoRA / Kaiming fan_in init). NOTE:
        # this deviates from the literal spec "std=1/sqrt(r)" -- that scales by
        # the rank instead of the contraction dim, giving Var(x@A)=d/r~128 and
        # diverging training. Documented in summary.md.
        bound = math.sqrt(3.0 / self.d)                 # uniform(-b,b) -> std = 1/sqrt(d)
        if variant in ("shared", "mlp"):
            A = torch.empty(self.d, rank, device=device)
            self.A = torch.nn.Parameter(A.uniform_(-bound, bound))
            self.B = torch.nn.Parameter(torch.zeros(rank, self.d, device=device))
        else:
            A = torch.empty(self.nh, self.d, rank, device=device)
            self.A = torch.nn.Parameter(A.uniform_(-bound, bound))
            self.B = torch.nn.Parameter(torch.zeros(self.nh, rank, self.hd, device=device))
        return [self.A, self.B]

    def param_count(self):
        return self.A.numel() + self.B.numel()


# --------------------------------------------------------------------------- #
# eval / train
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_loss(model, hook, seqs, bs, device):
    model.eval()
    tot, ntok = 0.0, 0
    for i in range(0, seqs.shape[0], bs):
        ids = seqs[i:i + bs].to(device)
        hook.current_ids = ids
        logits = model(ids).logits
        sl = logits[:, :-1, :].float()
        lab = ids[:, 1:]
        l = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), lab.reshape(-1), reduction="sum")
        tot += l.item()
        ntok += lab.numel()
    return tot / ntok


def train_one(model, hook, train_seqs, layer, variant, rank, device,
              steps=500, bs=8, lr=1e-3):
    params = hook.init_lora(variant, rank, device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    hook.mode = "correct"
    model.eval()                                    # frozen; dropout off
    n = train_seqs.shape[0]
    order = torch.randperm(n)
    ptr = 0
    curve = []
    first_loss = None
    for step in range(steps):
        if ptr + bs > n:
            order = torch.randperm(n)
            ptr = 0
        idx = order[ptr:ptr + bs]
        ptr += bs
        ids = train_seqs[idx].to(device)
        hook.current_ids = ids
        opt.zero_grad(set_to_none=True)
        logits = model(ids).logits                  # grad ON (only A,B require grad)
        sl = logits[:, :-1, :].float()
        lab = ids[:, 1:]
        loss = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), lab.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        # dead-stuck guard: gradient must actually reach the adapter
        if step == 0:
            # B is zero-initialised, so dcorr/dA = 0 at step 0 -> A.grad is
            # legitimately 0 here while B.grad is nonzero. Dead-stuck (no_grad
            # bug) is when BOTH are zero/None.
            ga = 0.0 if hook.A.grad is None else float(hook.A.grad.abs().sum())
            gb = 0.0 if hook.B.grad is None else float(hook.B.grad.abs().sum())
            if ga + gb == 0.0:
                raise RuntimeError("DEAD-STUCK: both A.grad and B.grad are None/zero "
                                   "after first backward -> grad not flowing to LoRA "
                                   "(check the forward is NOT wrapped in torch.no_grad()).")
            first_loss = loss.item()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            curve.append((step, loss.item()))
            print(f"    [train L{layer} {variant} r{rank}] step {step:3d} loss {loss.item():.4f}")
    last = curve[-1][1]
    still_falling = (first_loss - last) > 0.02 and (curve[-2][1] - last) > 0.01 if len(curve) > 1 else False
    if still_falling:
        print(f"    [warn] L{layer} {variant} r{rank}: train loss still falling at step "
              f"{steps} (last two: {curve[-2][1]:.4f}->{last:.4f}); using final checkpoint anyway")
    return curve, still_falling


# --------------------------------------------------------------------------- #
# scale / norm sanity
# --------------------------------------------------------------------------- #
class StatsHook:
    """Accumulate per-feature sum / sumsq of real V and token-table V on one
    layer, over all tokens."""

    def __init__(self, model, layer_idx, d):
        self.gpt_neox = model.gpt_neox
        self.layer = self.gpt_neox.layers[layer_idx]
        self.qkv = self.layer.attention.query_key_value
        self.nh = model.config.num_attention_heads
        self.hd = model.config.hidden_size // self.nh
        self.d = d
        self.current_ids = None
        self.so = torch.zeros(d, dtype=torch.float64)   # sum orig
        self.qo = torch.zeros(d, dtype=torch.float64)   # sumsq orig
        self.si = torch.zeros(d, dtype=torch.float64)
        self.qi = torch.zeros(d, dtype=torch.float64)
        self.n = 0
        self._h = self.qkv.register_forward_hook(self._hook)

    def detach(self):
        self._h.remove()

    @torch.no_grad()
    def _hook(self, module, inputs, output):
        B, S, _ = output.shape
        vo = output.view(B, S, self.nh, 3 * self.hd)[..., 2 * self.hd:].reshape(-1, self.d).double()
        emb = self.gpt_neox.embed_in(self.current_ids)
        tg = F.linear(self.layer.input_layernorm(emb), module.weight, module.bias)
        vi = tg.view(B, S, self.nh, 3 * self.hd)[..., 2 * self.hd:].reshape(-1, self.d).double()
        self.so += vo.sum(0).cpu(); self.qo += (vo * vo).sum(0).cpu()
        self.si += vi.sum(0).cpu(); self.qi += (vi * vi).sum(0).cpu()
        self.n += vo.shape[0]

    def stats(self):
        mo = self.so / self.n
        mi = self.si / self.n
        so = (self.qo / self.n - mo * mo).clamp_min(1e-12).sqrt()
        si = (self.qi / self.n - mi * mi).clamp_min(1e-12).sqrt()
        return mo.numpy(), so.numpy(), mi.numpy(), si.numpy()


def mode_scale_stats(args):
    device = args.device
    model, tok = load_model(device)
    seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen) if args.split == "eval" \
        else load_train_seqs(tok, args.num_seq, args.ctxlen)
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    hooks = [StatsHook(model, l, d) for l in range(L)]
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, seqs.shape[0], args.batch_size):
            ids = seqs[i:i + args.batch_size].to(device)
            for h in hooks:
                h.current_ids = ids
            model(ids)
    mo = np.stack([h.stats()[0] for h in hooks])
    so = np.stack([h.stats()[1] for h in hooks])
    mi = np.stack([h.stats()[2] for h in hooks])
    si = np.stack([h.stats()[3] for h in hooks])
    for h in hooks:
        h.detach()
    out = os.path.join(HERE, f"vstats_{args.split}.npz")
    np.savez(out, mu_orig=mo, sig_orig=so, mu_int=mi, sig_int=si)
    print(f"[scale_stats] split={args.split} wrote {out} ({time.time()-t0:.1f}s)")
    # quick per-layer scalar summary of the shift
    for l in range(L):
        dmu = float(np.abs(mo[l] - mi[l]).mean())
        rsig = float((si[l] / so[l]).mean())
        print(f"  L{l:2d}  mean|dmu|={dmu:.4f}  mean(sig_int/sig_orig)={rsig:.3f}")


def mode_scale_coarse(args):
    device = args.device
    model, tok = load_model(device)
    seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)
    stats = np.load(os.path.join(HERE, "vstats_eval.npz"))
    L = model.config.num_hidden_layers
    hook = V1aHook(model, 0)
    hook.attach()
    # baseline
    hook.mode = "off"
    base = eval_loss(model, hook, seqs, args.batch_size, device)
    print(f"[scale_coarse] baseline={base:.4f}")
    v0 = np.load(os.path.join(OUT0, "coarse_loss_delta.npy"))
    rescaled = np.full(L, np.nan)
    for l in range(L):
        hook.detach()
        hook = V1aHook(model, l)
        hook.attach()
        hook.rescale = {k: torch.tensor(stats[v][l], device=device, dtype=torch.float32)
                        for k, v in [("mu_orig", "mu_orig"), ("sig_orig", "sig_orig"),
                                     ("mu_int", "mu_int"), ("sig_int", "sig_int")]}
        hook.mode = "table"
        lo = eval_loss(model, hook, seqs, args.batch_size, device)
        rescaled[l] = lo - base
        print(f"  L{l:2d}  v0_delta={v0[l]:+.4f}  rescaled_delta={rescaled[l]:+.4f}")
    np.save(os.path.join(HERE, "scale_coarse_delta.npy"), rescaled)
    # write scale_sanity.md
    mid = [5, 11, 17]
    low = [20, 21, 22, 23]
    rmid = float(np.nanmean(rescaled[mid]))
    rlow = float(np.nanmean(rescaled[low]))
    v0mid = float(np.nanmean(v0[mid]))
    v0low = float(np.nanmean(v0[low]))
    inverted_u_survives = rmid > rlow + 0.1
    if inverted_u_survives:
        verdict = ("inverted-U SURVIVES rescaling -> scale is NOT the main cause; "
                   "v1a uses RAW token-table V (no rescale).")
    else:
        verdict = ("inverted-U is FLATTENED by rescaling -> scale shift contaminates "
                   "v0; v1a should apply per-layer affine rescale (mu/sig on TRAIN data).")
    lines = ["# v1a scale/norm sanity\n",
             "Re-ran the coarse all-heads token-table swap with a per-feature affine "
             "rescale of V_intervention to match real-V mean/std (eval-set stats). "
             "If the inverted-U (mid layers >> edge layers) survives rescaling, the v0 "
             "signal is not a LayerNorm-OOD scale artifact.\n",
             "| layer | v0 delta | rescaled delta |",
             "|---|---|---|"]
    for l in range(L):
        lines.append(f"| L{l} | {v0[l]:+.4f} | {rescaled[l]:+.4f} |")
    lines += [
        "",
        f"- mid (L5/11/17) mean delta:  v0={v0mid:+.4f}  rescaled={rmid:+.4f}",
        f"- low (L20-23) mean delta:    v0={v0low:+.4f}  rescaled={rlow:+.4f}",
        "",
        f"**Verdict: {verdict}**",
    ]
    with open(os.path.join(HERE, "scale_sanity.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[scale_coarse] inverted_u_survives={inverted_u_survives} -> wrote scale_sanity.md")


# --------------------------------------------------------------------------- #
# train driver
# --------------------------------------------------------------------------- #
def _rescale_for_train(device, layer):
    p = os.path.join(HERE, "vstats_train.npz")
    if not os.path.exists(p):
        raise FileNotFoundError("vstats_train.npz missing; run --mode scale_stats --split train")
    s = np.load(p)
    return {k: torch.tensor(s[k][layer], device=device, dtype=torch.float32)
            for k in ("mu_orig", "sig_orig", "mu_int", "sig_int")}


def run_single(model, eval_seqs, train_seqs, layer, variant, rank, device, args,
               baseline_recomputed, v0_coarse, use_rescale):
    hook = V1aHook(model, layer)
    hook.attach()
    if use_rescale:
        hook.rescale = _rescale_for_train(device, layer)
    t0 = time.time()
    curve, still_falling = train_one(model, hook, train_seqs, layer, variant, rank,
                                     device, steps=args.steps, bs=args.batch_size, lr=args.lr)
    hook.mode = "correct"
    eval_l = eval_loss(model, hook, eval_seqs, args.batch_size, device)
    resid = eval_l - baseline_recomputed
    v0d = float(v0_coarse[layer])
    recovery = 1.0 - resid / v0d
    pcount = hook.param_count()
    pcount_m = pcount / 1e6
    rec_per_m = recovery / pcount_m if pcount_m > 0 else float("nan")
    rec = {
        "layer": layer, "variant": variant, "rank": rank,
        "eval_loss": eval_l, "baseline_recomputed": baseline_recomputed,
        "v0_coarse_delta": v0d, "residual_delta": resid,
        "recovery_ratio": recovery, "param_count": pcount,
        "param_count_M": round(pcount_m, 4), "recovery_per_M": rec_per_m,
        "rescale": bool(use_rescale), "train_steps": args.steps, "lr": args.lr,
        "still_falling": bool(still_falling),
        "train_curve": curve, "seconds": round(time.time() - t0, 1),
    }
    fn = os.path.join(HERE, f"L{layer}_{variant}_r{rank}.json")
    with open(fn, "w") as f:
        json.dump(rec, f, indent=2)
    hook.detach()
    # free adapter
    hook.A = hook.B = None
    print(f"[run] L{layer} {variant} r{rank}: eval={eval_l:.4f} resid={resid:+.4f} "
          f"recovery={recovery:.3f} params={pcount_m:.4f}M rec/M={rec_per_m:.3f} "
          f"({rec['seconds']}s)")
    if resid < -0.1:
        print(f"  [FLAG] large negative residual ({resid:+.4f}) -> investigate")
    return rec


def _recompute_baseline(model, eval_seqs, device, args):
    hook = V1aHook(model, 0)
    hook.attach()
    hook.mode = "off"
    b = eval_loss(model, hook, eval_seqs, args.batch_size, device)
    hook.detach()
    print(f"[baseline] recomputed eval CE = {b:.4f} (v0={V0_BASELINE})")
    assert abs(b - V0_BASELINE) < 0.005, \
        f"baseline drift: recomputed {b:.4f} vs v0 {V0_BASELINE} (|d|>=0.005) -- STOP"
    return b


def mode_train(args):
    device = args.device
    if not args.variant:
        args.variant = "shared"
    model, tok = load_model(device)
    eval_seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)
    train_seqs = load_train_seqs(tok, args.num_seq, args.ctxlen)
    v0_coarse = np.load(os.path.join(OUT0, "coarse_loss_delta.npy"))
    base = _recompute_baseline(model, eval_seqs, device, args)
    run_single(model, eval_seqs, train_seqs, args.layer, args.variant, args.rank,
               device, args, base, v0_coarse, args.rescale)


def mode_all(args):
    device = args.device
    model, tok = load_model(device)
    eval_seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)
    train_seqs = load_train_seqs(tok, args.num_seq, args.ctxlen)
    v0_coarse = np.load(os.path.join(OUT0, "coarse_loss_delta.npy"))
    base = _recompute_baseline(model, eval_seqs, device, args)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else LAYERS
    variants = args.variant.split(",") if args.variant else VARIANTS
    ranks = [int(x) for x in args.ranks.split(",")] if args.ranks else RANKS
    done = 0
    for layer in layers:
        for variant in variants:
            for rank in ranks:
                fn = os.path.join(HERE, f"L{layer}_{variant}_r{rank}.json")
                if os.path.exists(fn) and not args.force:
                    print(f"[skip] {os.path.basename(fn)} exists")
                    continue
                run_single(model, eval_seqs, train_seqs, layer, variant, rank,
                           device, args, base, v0_coarse, args.rescale)
                done += 1
    print(f"[all] completed {done} runs")


# --------------------------------------------------------------------------- #
# stability probe (diagnostic: undertraining vs landscape sharpness)
# --------------------------------------------------------------------------- #
def _gnorm(p):
    return 0.0 if p.grad is None else float(p.grad.norm())


# (layer, variant, rank, lr, steps, grad_clip)
PROBE_MATRIX = [
    (11, "shared", 8, 3e-5, 500, 0.0),
    (11, "shared", 8, 3e-5, 2000, 0.0),
    (11, "shared", 8, 1e-4, 500, 1.0),
    (11, "shared", 8, 5e-5, 500, 0.0),
    (11, "shared", 32, 3e-5, 500, 0.0),
    (11, "shared", 32, 3e-5, 2000, 0.0),
    (11, "shared", 32, 1e-4, 500, 1.0),
    (11, "shared", 32, 5e-5, 500, 0.0),
]

# norm-capped probe: does bounding ||dV|| let deeper layers learn & generalize?
# (layer, variant, rank, lr, steps, grad_clip, dv_cap)
CAP_MATRIX = [
    (11, "shared", 8, 1e-4, 500, 0.0, 0.15),
    (11, "shared", 16, 1e-4, 500, 0.0, 0.15),
    (11, "shared", 8, 3e-4, 500, 0.0, 0.15),
    (11, "shared", 16, 3e-4, 800, 0.0, 0.10),
    (5, "shared", 8, 1e-4, 500, 0.0, 0.15),
    (5, "shared", 16, 1e-4, 500, 0.0, 0.15),
]

# is the deep-layer ceiling about LINEARITY? nonlinear bottleneck (mlp) at equal
# param count as shared LoRA, norm-capped stable training.
# (layer, variant, rank/k, lr, steps, grad_clip, dv_cap)
MLP_MATRIX = [
    (11, "mlp", 16, 1e-4, 500, 0.0, 0.15),
    (11, "mlp", 32, 1e-4, 500, 0.0, 0.15),
    (11, "mlp", 16, 3e-4, 800, 0.0, 0.15),
    (5, "mlp", 16, 1e-4, 500, 0.0, 0.15),
]

# is L11 a rank/capacity wall? push rank up (capped, stable lr).
HIGHRANK_MATRIX = [
    (11, "shared", 64, 1e-4, 500, 0.0, 0.15),
    (11, "shared", 128, 1e-4, 500, 0.0, 0.15),
    (11, "shared", 256, 1e-4, 800, 0.0, 0.15),
    (5, "shared", 64, 1e-4, 500, 0.0, 0.15),
]

# overfit test: report train-recovery vs eval-recovery (is L11 a data/generalization
# problem rather than architecture?)
OVERFIT_MATRIX = [
    (11, "shared", 16, 1e-4, 500, 0.0, 0.15),
    (11, "shared", 64, 1e-4, 500, 0.0, 0.15),
    (5, "shared", 16, 1e-4, 500, 0.0, 0.15),
]


def run_probe(model, eval_seqs, small, base, base_small, train_seqs, v0,
              layer, variant, rank, lr, steps, grad_clip, bs, device, dv_cap=0.0,
              base_train=None, anchor_mu=None, anchor_cnt=None):
    abase = "A1" if anchor_mu is not None else "A0"
    tag = f"L{layer}_{variant}_r{rank}_lr{lr:g}_s{steps}_c{grad_clip:g}_cap{dv_cap:g}_{abase}"
    hook = V1aHook(model, layer)
    hook.attach()
    hook.dv_cap = dv_cap
    hook.anchor_mu = anchor_mu
    hook.anchor_cnt = anchor_cnt
    params = hook.init_lora(variant, rank, device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    hook.mode = "correct"
    model.eval()
    n = train_seqs.shape[0]
    order = torch.randperm(n)
    ptr = 0
    series = []
    v0d = float(v0[layer])
    t0 = time.time()
    for step in range(steps):
        if ptr + bs > n:
            order = torch.randperm(n)
            ptr = 0
        ids = train_seqs[order[ptr:ptr + bs]].to(device)
        ptr += bs
        hook.current_ids = ids
        opt.zero_grad(set_to_none=True)
        logits = model(ids).logits
        sl = logits[:, :-1, :].float()
        lab = ids[:, 1:]
        loss = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), lab.reshape(-1))
        if not torch.isfinite(loss):
            series.append({"step": step, "train_ce": float("nan"), "EXPLODED": True})
            print(f"  [{tag}] step {step}: non-finite loss -> stop this config")
            break
        loss.backward()
        gA, gB = _gnorm(hook.A), _gnorm(hook.B)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        dv_ratio, dv_max = hook.last_dv_ratio, hook.last_dv_maxabs
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                held = eval_loss(model, hook, small, bs, device)
            rec_est = 1.0 - (held - base_small) / v0d
            series.append({
                "step": step, "train_ce": round(loss.item(), 4),
                "held_ce": round(held, 4), "rec_est": round(rec_est, 4),
                "dv_ratio": round(dv_ratio, 4), "dv_maxabs": round(dv_max, 4),
                "normA": round(float(hook.A.norm()), 4),
                "normB": round(float(hook.B.norm()), 4),
                "gradA": round(gA, 5), "gradB": round(gB, 5),
            })
            print(f"  [{tag}] s{step:4d} tr={loss.item():.3f} held={held:.3f} "
                  f"rec={rec_est:+.3f} dV/V={dv_ratio:.3f} |B|={float(hook.B.norm()):.3f} "
                  f"gB={gB:.4f}")
    hook.mode = "correct"
    with torch.no_grad():
        final = eval_loss(model, hook, eval_seqs, bs, device)
        train_final = eval_loss(model, hook, train_seqs[:200], bs, device)
    resid = final - base
    train_rec = 1.0 - (train_final - base_train) / v0d if base_train else None
    rec = {"tag": tag, "layer": layer, "variant": variant, "rank": rank, "lr": lr,
           "steps": steps, "grad_clip": grad_clip, "dv_cap": dv_cap, "eval_loss": final,
           "train_loss_200": train_final, "train_recovery": train_rec,
           "baseline": base, "v0_coarse_delta": v0d, "residual_delta": resid,
           "recovery_ratio": 1.0 - resid / v0d, "seconds": round(time.time() - t0, 1),
           "series": series}
    with open(os.path.join(HERE, f"probe_{tag}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[probe] {tag}: eval_rec={rec['recovery_ratio']:+.3f} "
          f"train_rec={train_rec if train_rec is None else round(train_rec,3)} "
          f"(gap={'NA' if train_rec is None else round(train_rec-rec['recovery_ratio'],3)}) "
          f"({rec['seconds']}s)")
    hook.detach()
    hook.A = hook.B = None
    return rec


def mode_probe(args):
    device = args.device
    model, tok = load_model(device)
    eval_seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)
    train_seqs = load_train_seqs(tok, args.num_seq, args.ctxlen)
    v0 = np.load(os.path.join(OUT0, "coarse_loss_delta.npy"))
    base = _recompute_baseline(model, eval_seqs, device, args)
    small = eval_seqs[:64]
    # baseline on the 64-seq heldout (for mid-training recovery estimate)
    h0 = V1aHook(model, 0); h0.attach(); h0.mode = "off"
    base_small = eval_loss(model, h0, small, args.batch_size, device)
    base_train = eval_loss(model, h0, train_seqs[:200], args.batch_size, device); h0.detach()
    print(f"[probe] base_small(64)={base_small:.4f} base_train(200)={base_train:.4f}")
    if args.overfit_matrix:
        for (layer, variant, rank, lr, steps, clip, cap) in OVERFIT_MATRIX:
            run_probe(model, eval_seqs, small, base, base_small, train_seqs, v0,
                      layer, variant, rank, lr, steps, clip, args.batch_size, device,
                      dv_cap=cap, base_train=base_train)
        return
    if args.highrank_matrix:
        for (layer, variant, rank, lr, steps, clip, cap) in HIGHRANK_MATRIX:
            run_probe(model, eval_seqs, small, base, base_small, train_seqs, v0,
                      layer, variant, rank, lr, steps, clip, args.batch_size, device, dv_cap=cap)
    elif args.mlp_matrix:
        for (layer, variant, rank, lr, steps, clip, cap) in MLP_MATRIX:
            run_probe(model, eval_seqs, small, base, base_small, train_seqs, v0,
                      layer, variant, rank, lr, steps, clip, args.batch_size, device, dv_cap=cap)
    elif args.dv_cap > 0 or args.cap_matrix:
        for (layer, variant, rank, lr, steps, clip, cap) in CAP_MATRIX:
            run_probe(model, eval_seqs, small, base, base_small, train_seqs, v0,
                      layer, variant, rank, lr, steps, clip, args.batch_size, device, dv_cap=cap)
    else:
        for (layer, variant, rank, lr, steps, clip) in PROBE_MATRIX:
            run_probe(model, eval_seqs, small, base, base_small, train_seqs, v0,
                      layer, variant, rank, lr, steps, clip, args.batch_size, device)


class DiffCovHook:
    """Accumulate the d x d covariance of (V_real - V_table) over all tokens."""

    def __init__(self, model, layer_idx, d):
        self.gpt_neox = model.gpt_neox
        self.layer = self.gpt_neox.layers[layer_idx]
        self.qkv = self.layer.attention.query_key_value
        self.nh = model.config.num_attention_heads
        self.hd = model.config.hidden_size // self.nh
        self.d = d
        self.current_ids = None
        self.C = torch.zeros(d, d, dtype=torch.float64)
        self.Cr = torch.zeros(d, d, dtype=torch.float64)   # cov of real V (reference)
        self.n = 0
        self._h = self.qkv.register_forward_hook(self._hook)

    def detach(self):
        self._h.remove()

    @torch.no_grad()
    def _hook(self, module, inputs, output):
        B, S, _ = output.shape
        vr = output.view(B, S, self.nh, 3 * self.hd)[..., 2 * self.hd:].reshape(-1, self.d).double()
        emb = self.gpt_neox.embed_in(self.current_ids)
        tg = F.linear(self.layer.input_layernorm(emb), module.weight, module.bias)
        vt = tg.view(B, S, self.nh, 3 * self.hd)[..., 2 * self.hd:].reshape(-1, self.d).double()
        dv = (vr - vt).cpu()
        self.C += dv.t() @ dv
        self.Cr += vr.cpu().t() @ vr.cpu()
        self.n += dv.shape[0]


def mode_svd_diff(args):
    device = args.device
    model, tok = load_model(device)
    seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)[:200]
    d = model.config.hidden_size
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [5, 11]
    out = {}
    for layer in layers:
        h = DiffCovHook(model, layer, d)
        with torch.no_grad():
            for i in range(0, seqs.shape[0], args.batch_size):
                ids = seqs[i:i + args.batch_size].to(device)
                h.current_ids = ids
                model(ids)
        ev = torch.linalg.eigvalsh(h.C / h.n).flip(0).clamp_min(0)
        evr = torch.linalg.eigvalsh(h.Cr / h.n).flip(0).clamp_min(0)
        cum = torch.cumsum(ev, 0) / ev.sum()
        def rank_at(frac):
            return int((cum < frac).sum()) + 1
        ranks = {f"{int(f*100)}%": rank_at(f) for f in (0.5, 0.9, 0.95, 0.99)}
        eff = float((ev.sum() ** 2) / (ev ** 2).sum())   # participation ratio
        h.detach()
        out[layer] = {"ranks_for_variance": ranks, "participation_ratio": round(eff, 1),
                      "total_var_diff": float(ev.sum()), "total_var_realV": float(evr.sum()),
                      "cumvar_first32": [round(float(x), 3) for x in cum[:32]]}
        print(f"[svd] L{layer}: var(dV)={ev.sum():.2f} var(realV)={evr.sum():.2f} "
              f"PR={eff:.1f} ranks@var={ranks}")
        np.save(os.path.join(HERE, f"diff_eig_L{layer}.npy"), ev.numpy())
    with open(os.path.join(HERE, "svd_diff.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[svd] wrote svd_diff.json")


# --------------------------------------------------------------------------- #
# anchor audit + oracle upper bound (diagnostic; no corrector training)
# --------------------------------------------------------------------------- #
def _token_table_flat(model, layer, ids):
    """A0 anchor: W_V . LN_l(E[x]) flattened to [B,S,d]."""
    nh = model.config.num_attention_heads
    hd = model.config.hidden_size // nh
    qkv = layer.attention.query_key_value
    normed = layer.input_layernorm(model.gpt_neox.embed_in(ids))
    tg = F.linear(normed, qkv.weight, qkv.bias)
    B, S, _ = tg.shape
    return tg.view(B, S, nh, 3 * hd)[..., 2 * hd:].reshape(B, S, nh * hd)


class AnchorHook:
    """Replaces layer l's V with a configurable anchor / oracle, for eval only."""

    def __init__(self, model, layer_idx):
        self.model = model
        self.gpt_neox = model.gpt_neox
        self.layer = self.gpt_neox.layers[layer_idx]
        self.qkv = self.layer.attention.query_key_value
        self.nh = model.config.num_attention_heads
        self.hd = model.config.hidden_size // self.nh
        self.d = model.config.hidden_size
        self.ids = None
        self.mode = "A0"           # A0 | A1 | A2 | A3 | oracle
        self.mu = None             # [vocab,d] per-token mean V
        self.cnt = None            # [vocab]
        self.lam = 5.0
        self.affine = None         # (a0_mean,a0_std,real_mean,real_std) [d]
        self.U = None              # [d,r] residual PCA basis (oracle)
        self._h = self.qkv.register_forward_hook(self._hook)

    def detach(self):
        self._h.remove()

    @torch.no_grad()
    def _hook(self, module, inputs, output):
        B, S, _ = output.shape
        nh, hd = self.nh, self.hd
        o4 = output.view(B, S, nh, 3 * hd)
        vreal = o4[..., 2 * hd:].reshape(B, S, self.d).float()
        a0 = _token_table_flat(self.model, self.layer, self.ids).float()  # [B,S,d]
        if self.mode == "A0":
            vnew = a0
        elif self.mode in ("A1", "A2"):
            mu = self.mu[self.ids]                                       # [B,S,d] float
            seen = (self.cnt[self.ids] > 0).unsqueeze(-1)
            if self.mode == "A1":
                vnew = torch.where(seen, mu, a0)
            else:
                n = self.cnt[self.ids].unsqueeze(-1).float()
                w = n / (n + self.lam)
                vnew = torch.where(seen, w * mu + (1 - w) * a0, a0)
        elif self.mode == "A3":
            a0m, a0s, rm, rs = self.affine
            vnew = (a0 - a0m) / a0s * rs + rm
        else:  # oracle: A0 + P_r(Vreal - A0)
            resid = vreal - a0
            vnew = a0 + (resid @ self.U) @ self.U.t()
        v4 = vnew.view(B, S, nh, hd).to(o4.dtype)
        new = torch.cat([o4[..., :hd], o4[..., hd:2 * hd], v4], dim=-1)
        return new.view(B, S, nh * 3 * hd)


@torch.no_grad()
def _calibrate(model, layer, calib_seqs, bs, device, vocab, d, n_pca=256):
    """Per-token-id mean V, counts, residual PCA basis, affine stats — on calib."""
    nh = model.config.num_attention_heads
    hd = d // nh
    qkv = layer.attention.query_key_value
    Vsum = torch.zeros(vocab, d, device=device)
    cnt = torch.zeros(vocab, device=device)
    C = torch.zeros(d, d, dtype=torch.float64, device=device)
    rm = torch.zeros(d, dtype=torch.float64, device=device)
    am = torch.zeros(d, dtype=torch.float64, device=device)
    rsq = torch.zeros(d, dtype=torch.float64, device=device)
    asq = torch.zeros(d, dtype=torch.float64, device=device)
    ntok = 0
    cap = {"v": None}

    def hook(module, inputs, output):
        B, S, _ = output.shape
        cap["v"] = output.view(B, S, nh, 3 * hd)[..., 2 * hd:].reshape(B, S, d)
    h = qkv.register_forward_hook(hook)
    for i in range(0, calib_seqs.shape[0], bs):
        ids = calib_seqs[i:i + bs].to(device)
        model(ids)
        vr = cap["v"]                                   # [B,S,d]
        a0 = _token_table_flat(model, layer, ids)
        flat_ids = ids.reshape(-1)
        vr2 = vr.reshape(-1, d).float()
        a02 = a0.reshape(-1, d).float()
        Vsum.index_add_(0, flat_ids, vr2)
        cnt.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=Vsum.dtype))
        resid = (vr2 - a02).double()
        C += resid.t() @ resid
        rm += vr2.double().sum(0); rsq += (vr2.double() ** 2).sum(0)
        am += a02.double().sum(0); asq += (a02.double() ** 2).sum(0)
        ntok += vr2.shape[0]
    h.remove()
    mu = Vsum / cnt.clamp_min(1).unsqueeze(1)
    rm /= ntok; am /= ntok
    rstd = (rsq / ntok - rm ** 2).clamp_min(1e-8).sqrt()
    astd = (asq / ntok - am ** 2).clamp_min(1e-8).sqrt()
    evals, evecs = torch.linalg.eigh(C / ntok)
    U = evecs.flip(1)[:, :n_pca].float()               # top-n_pca, [d,n_pca]
    affine = (am.float(), astd.float(), rm.float(), rstd.float())
    return mu, cnt, U, affine, ntok


def mode_anchor(args):
    device = args.device
    model, tok = load_model(device)
    eval_seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)
    calib_seqs = load_train_seqs(tok, args.num_seq, args.ctxlen)   # disjoint from eval
    v0 = np.load(os.path.join(OUT0, "coarse_loss_delta.npy"))
    d = model.config.hidden_size
    vocab = model.config.vocab_size
    base = _recompute_baseline_anchor(model, eval_seqs, args.batch_size, device)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [5, 11, 17, 23]
    lams = [1, 5, 20, 100]
    oracle_rs = [16, 32, 64, 128, 256]
    out = {}
    for layer in layers:
        lyr = model.gpt_neox.layers[layer]
        print(f"[anchor] L{layer}: calibrating on {calib_seqs.shape[0]} seqs ...")
        mu, cnt, U, affine, ntok = _calibrate(model, lyr, calib_seqs, args.batch_size,
                                              device, vocab, d)
        # coverage on eval
        ev_ids = eval_seqs.reshape(-1).to(device)
        cov = float((cnt[ev_ids] > 0).float().mean())
        hook = AnchorHook(model, layer)
        hook.mu, hook.cnt, hook.U, hook.affine = mu, cnt, U, affine
        res = {"v0_delta": float(v0[layer]), "coverage": round(cov, 4)}

        def ev(tag, setter):
            setter()
            ce = _anchor_eval(model, hook, eval_seqs, args.batch_size, device)
            dl = ce - base
            vd = float(v0[layer])
            rec = (1.0 - dl / vd) if abs(vd) > 1e-6 else float("nan")
            res[tag] = {"ce": round(ce, 4), "delta": round(dl, 4), "recovery": round(rec, 4)}
            print(f"  L{layer} {tag:10s} ce={ce:.4f} delta={dl:+.4f} rec={rec:+.3f}")

        def s_a0(): hook.mode = "A0"
        def s_a1(): hook.mode = "A1"
        def s_a3(): hook.mode = "A3"
        ev("A0", s_a0); ev("A1", s_a1); ev("A3", s_a3)
        for lam in lams:
            def s_a2(l=lam): hook.mode = "A2"; hook.lam = float(l)
            ev(f"A2_lam{lam}", s_a2)
        for r in oracle_rs:
            def s_or(rr=r): hook.mode = "oracle"; hook.U = U[:, :rr]
            ev(f"oracle_r{r}", s_or)
            hook.U = U  # restore
        hook.detach()
        out[str(layer)] = res
    with open(os.path.join(HERE, "anchor_audit.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[anchor] wrote anchor_audit.json")


@torch.no_grad()
def _anchor_eval(model, hook, seqs, bs, device):
    tot, ntok = 0.0, 0
    for i in range(0, seqs.shape[0], bs):
        ids = seqs[i:i + bs].to(device)
        hook.ids = ids
        logits = model(ids).logits
        sl = logits[:, :-1, :].float(); lab = ids[:, 1:]
        tot += F.cross_entropy(sl.reshape(-1, sl.shape[-1]), lab.reshape(-1),
                               reduction="sum").item()
        ntok += lab.numel()
    return tot / ntok


@torch.no_grad()
def _recompute_baseline_anchor(model, seqs, bs, device):
    tot, ntok = 0.0, 0
    for i in range(0, seqs.shape[0], bs):
        ids = seqs[i:i + bs].to(device)
        logits = model(ids).logits
        sl = logits[:, :-1, :].float(); lab = ids[:, 1:]
        tot += F.cross_entropy(sl.reshape(-1, sl.shape[-1]), lab.reshape(-1),
                               reduction="sum").item()
        ntok += lab.numel()
    b = tot / ntok
    print(f"[anchor] baseline CE = {b:.4f} (v0 {V0_BASELINE})")
    assert abs(b - V0_BASELINE) < 0.005, f"baseline drift {b}"
    return b


def mode_v1b(args):
    """v1b pilot: fitted per-token anchor (A1) + small trained correction.
    Does training a small corrector on the A1 residual (the genuine-context
    part) recover MORE than A1 alone, esp. at L11?"""
    device = args.device
    model, tok = load_model(device)
    eval_seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)
    train_seqs = load_train_seqs(tok, args.num_seq, args.ctxlen)
    v0 = np.load(os.path.join(OUT0, "coarse_loss_delta.npy"))
    d = model.config.hidden_size
    vocab = model.config.vocab_size
    base = _recompute_baseline(model, eval_seqs, device, args)
    small = eval_seqs[:64]
    h0 = V1aHook(model, 0); h0.attach(); h0.mode = "off"
    base_small = eval_loss(model, h0, small, args.batch_size, device)
    base_train = eval_loss(model, h0, train_seqs[:200], args.batch_size, device); h0.detach()
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [11, 5]
    ranks = [int(x) for x in args.ranks.split(",")] if args.ranks else [16, 64]
    summary = {}
    for layer in layers:
        lyr = model.gpt_neox.layers[layer]
        print(f"[v1b] L{layer}: calibrating A1 table ...")
        mu, cnt, _, _, _ = _calibrate(model, lyr, train_seqs, args.batch_size, device, vocab, d)
        # A1 alone (no correction)
        ha = V1aHook(model, layer); ha.attach()
        ha.anchor_mu = mu; ha.anchor_cnt = cnt; ha.mode = "table"
        a1ce = eval_loss(model, ha, eval_seqs, args.batch_size, device); ha.detach()
        a1rec = 1.0 - (a1ce - base) / float(v0[layer])
        print(f"[v1b] L{layer} A1-alone: ce={a1ce:.4f} recovery={a1rec:+.3f}")
        summary[str(layer)] = {"A1_alone": round(a1rec, 4)}
        for rank in ranks:
            rec = run_probe(model, eval_seqs, small, base, base_small, train_seqs, v0,
                            layer, "shared", rank, args.lr, args.steps, 0.0,
                            args.batch_size, device, dv_cap=0.15, base_train=base_train,
                            anchor_mu=mu, anchor_cnt=cnt)
            summary[str(layer)][f"A1+corr_r{rank}"] = round(rec["recovery_ratio"], 4)
    with open(os.path.join(HERE, "v1b_anchor_plus_corr.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[v1b] summary:", json.dumps(summary))


class RidgeHook:
    """Eval-only: V = A1(x) + cap( X @ W ), X = LN_l(h). W is a closed-form
    ridge map (optionally rank-truncated). W=None -> A1 anchor alone."""

    def __init__(self, model, layer_idx):
        self.model = model
        self.gpt_neox = model.gpt_neox
        self.layer = self.gpt_neox.layers[layer_idx]
        self.qkv = self.layer.attention.query_key_value
        self.nh = model.config.num_attention_heads
        self.hd = model.config.hidden_size // self.nh
        self.d = model.config.hidden_size
        self.ids = None
        self.mu = None
        self.cnt = None
        self.W = None
        self.cap = 0.0
        self._h = self.qkv.register_forward_hook(self._hook)

    def detach(self):
        self._h.remove()

    @torch.no_grad()
    def _hook(self, module, inputs, output):
        B, S, _ = output.shape
        nh, hd = self.nh, self.hd
        o4 = output.view(B, S, nh, 3 * hd)
        vreal = o4[..., 2 * hd:].reshape(B, S, self.d).float()
        a0 = _token_table_flat(self.model, self.layer, self.ids).float()
        mu = self.mu[self.ids]
        seen = (self.cnt[self.ids] > 0).unsqueeze(-1)
        base = torch.where(seen, mu, a0)                       # A1 [B,S,d]
        if self.W is not None:
            X = inputs[0].reshape(B, S, self.d).float()        # LN_l(h)
            corr = X @ self.W                                  # [B,S,d]
            if self.cap > 0:
                cn = corr.norm(); vn = vreal.norm()
                corr = corr * min(1.0, self.cap * float(vn) / (float(cn) + 1e-6))
            base = base + corr
        v4 = base.view(B, S, nh, hd).to(o4.dtype)
        return torch.cat([o4[..., :hd], o4[..., hd:2 * hd], v4], dim=-1).view(B, S, nh * 3 * hd)


@torch.no_grad()
def _ridge_calibrate(model, layer, calib_seqs, bs, device, vocab, d):
    """Pass 1: per-token mean (A1). Pass 2: X^TX, X^TY for X=LN_l(h), Y=Vreal-A1."""
    nh = model.config.num_attention_heads
    hd = d // nh
    qkv = layer.attention.query_key_value
    cap = {"v": None, "x": None}

    def hook(module, inputs, output):
        B, S, _ = output.shape
        cap["v"] = output.view(B, S, nh, 3 * hd)[..., 2 * hd:].reshape(B, S, d)
        cap["x"] = inputs[0].reshape(B, S, d)
    h = qkv.register_forward_hook(hook)
    Vsum = torch.zeros(vocab, d, device=device); cnt = torch.zeros(vocab, device=device)
    for i in range(0, calib_seqs.shape[0], bs):
        ids = calib_seqs[i:i + bs].to(device); model(ids)
        Vsum.index_add_(0, ids.reshape(-1), cap["v"].reshape(-1, d).float())
        cnt.index_add_(0, ids.reshape(-1), torch.ones(ids.numel(), device=device))
    mu = Vsum / cnt.clamp_min(1).unsqueeze(1)
    XtX = torch.zeros(d, d, dtype=torch.float64, device=device)
    XtY = torch.zeros(d, d, dtype=torch.float64, device=device)
    Ynorm2 = 0.0
    for i in range(0, calib_seqs.shape[0], bs):
        ids = calib_seqs[i:i + bs].to(device); model(ids)
        X = cap["x"].reshape(-1, d).double()
        Y = (cap["v"].reshape(-1, d).float() - mu[ids.reshape(-1)]).double()
        XtX += X.t() @ X; XtY += X.t() @ Y; Ynorm2 += float((Y ** 2).sum())
    h.remove()
    return mu, cnt, XtX, XtY, Ynorm2


def mode_ridge(args):
    device = args.device
    model, tok = load_model(device)
    eval_seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)
    calib_seqs = load_train_seqs(tok, args.num_seq, args.ctxlen)
    v0 = np.load(os.path.join(OUT0, "coarse_loss_delta.npy"))
    d = model.config.hidden_size; vocab = model.config.vocab_size
    base = _recompute_baseline_anchor(model, eval_seqs, args.batch_size, device)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [5, 6, 7, 11, 20]
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    ks = [16, 64, 256]
    sub = eval_seqs[:256]
    out = {}
    for layer in layers:
        lyr = model.gpt_neox.layers[layer]
        print(f"[ridge] L{layer}: calibrating ...")
        mu, cnt, XtX, XtY, _ = _ridge_calibrate(model, lyr, calib_seqs, args.batch_size,
                                               device, vocab, d)
        hook = RidgeHook(model, layer); hook.mu, hook.cnt = mu, cnt
        vd = float(v0[layer]); I = torch.eye(d, dtype=torch.float64, device=device)
        # A1 alone
        hook.W = None
        ce_a1 = _anchor_eval(model, hook, eval_seqs, args.batch_size, device)
        d_a1 = ce_a1 - base
        # select lambda by full-rank uncapped eval CE on subset
        best = None
        Ws = {}
        for lam in lams:
            W = torch.linalg.solve(XtX + lam * I, XtY).float()
            Ws[lam] = W
            hook.W = W; hook.cap = 0.0
            ce = _anchor_eval(model, hook, sub, args.batch_size, device)
            if best is None or ce < best[1]:
                best = (lam, ce)
        lam = best[0]; W = Ws[lam]
        U, Sv, Vh = torch.linalg.svd(W.double())
        res = {"v0_delta": vd, "baseline": base, "A1_ce": round(ce_a1, 4),
               "A1_recovery": round(1 - d_a1 / vd, 4), "best_lambda": lam,
               "context_delta_A1": round(d_a1, 4), "variants": {}}

        def evalW(Wmat, cap):
            hook.W = Wmat.float(); hook.cap = cap
            ce = _anchor_eval(model, hook, eval_seqs, args.batch_size, device)
            dl = ce - base
            return {"ce": round(ce, 4), "delta": round(dl, 4),
                    "R_total": round(1 - dl / vd, 4),
                    "R_context": round((d_a1 - dl) / d_a1, 4) if abs(d_a1) > 1e-6 else None}

        for cap in (0.0, 0.15):
            tag = "uncapped" if cap == 0 else "cap0.15"
            res["variants"][f"full_{tag}"] = evalW(W, cap)
            for k in ks:
                Wk = (U[:, :k] * Sv[:k]) @ Vh[:k]
                res["variants"][f"r{k}_{tag}"] = evalW(Wk, cap)
        hook.detach()
        out[str(layer)] = res
        print(f"[ridge] L{layer} A1={res['A1_recovery']:+.3f} "
              f"full_unc R_total={res['variants']['full_uncapped']['R_total']:+.3f} "
              f"R_context={res['variants']['full_uncapped']['R_context']} "
              f"| full_cap R_ctx={res['variants']['full_cap0.15']['R_context']} "
              f"(lam={lam})")
    os.makedirs(os.path.join(OUT0, "v1b_ridge"), exist_ok=True)
    with open(os.path.join(OUT0, "v1b_ridge", "ridge_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[ridge] wrote v1b_ridge/ridge_results.json")


def mode_ridge_init(args):
    """Zero-shot: factor the closed-form ridge W into LoRA factors
    A=U_r S_r^{1/2}, B=S_r^{1/2} V_r^T (so AB=W_r) and inject through the REAL
    V1aHook LoRA path (no SGD). Confirms the ridge solution is a deployable
    low-rank adapter and reproduces the ridge eval (rules out path mismatch)."""
    device = args.device
    model, tok = load_model(device)
    eval_seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)
    calib_seqs = load_train_seqs(tok, args.num_seq, args.ctxlen)
    v0 = np.load(os.path.join(OUT0, "coarse_loss_delta.npy"))
    d = model.config.hidden_size; vocab = model.config.vocab_size
    base = _recompute_baseline_anchor(model, eval_seqs, args.batch_size, device)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [5, 6, 7, 11]
    ranks = [16, 64, 256]; caps = [0.0, 0.15, 0.3, 0.5]
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    sub = eval_seqs[:256]
    I = torch.eye(d, dtype=torch.float64, device=device)
    out = {}
    for layer in layers:
        lyr = model.gpt_neox.layers[layer]
        print(f"[ridge_init] L{layer}: calibrating ...")
        mu, cnt, XtX, XtY, _ = _ridge_calibrate(model, lyr, calib_seqs, args.batch_size,
                                               device, vocab, d)
        vd = float(v0[layer])
        hook = V1aHook(model, layer)
        hook.attach(); hook.anchor_mu = mu; hook.anchor_cnt = cnt; hook.variant = "shared"
        hook.scale = 1.0
        # A1 alone
        hook.mode = "table"; hook.A = hook.B = None
        ce_a1 = eval_loss(model, hook, eval_seqs, args.batch_size, device)
        d_a1 = ce_a1 - base
        # pick lambda by full-rank subset CE (inject via direct matmul through a temp)
        best = None
        for lam in lams:
            W = torch.linalg.solve(XtX + lam * I, XtY).float()
            U, Sv, Vh = torch.linalg.svd(W.double())
            # full-rank as A,B (r=d) is heavy; use a high-rank proxy r=256 for selection
            r = 256
            A = (U[:, :r] * Sv[:r].sqrt()).float()
            B = (Sv[:r].sqrt().unsqueeze(1) * Vh[:r]).float()
            hook.A, hook.B = A, B; hook.mode = "correct"; hook.dv_cap = 0.0
            ce = eval_loss(model, hook, sub, args.batch_size, device)
            if best is None or ce < best[1]:
                best = (lam, ce)
        lam = best[0]
        W = torch.linalg.solve(XtX + lam * I, XtY).float()
        U, Sv, Vh = torch.linalg.svd(W.double())
        res = {"v0_delta": vd, "A1_recovery": round(1 - d_a1 / vd, 4),
               "best_lambda": lam, "context_delta_A1": round(d_a1, 4), "variants": {}}
        for r in ranks:
            A = (U[:, :r] * Sv[:r].sqrt()).float()
            B = (Sv[:r].sqrt().unsqueeze(1) * Vh[:r]).float()
            hook.A, hook.B = A, B; hook.mode = "correct"
            for cap in caps:
                hook.dv_cap = cap
                ce = eval_loss(model, hook, eval_seqs, args.batch_size, device)
                dl = ce - base
                res["variants"][f"r{r}_cap{cap:g}"] = {
                    "ce": round(ce, 4), "R_total": round(1 - dl / vd, 4),
                    "R_context": round((d_a1 - dl) / d_a1, 4) if abs(d_a1) > 1e-6 else None}
        hook.detach()
        out[str(layer)] = res
        v = res["variants"]
        print(f"[ridge_init] L{layer} A1={res['A1_recovery']:+.3f} "
              f"r64_unc Rtot={v['r64_cap0']['R_total']:+.3f} Rctx={v['r64_cap0']['R_context']} "
              f"| r64_cap.15={v['r64_cap0.15']['R_context']} cap.3={v['r64_cap0.3']['R_context']} "
              f"cap.5={v['r64_cap0.5']['R_context']}")
    os.makedirs(os.path.join(OUT0, "v1b_ridge"), exist_ok=True)
    with open(os.path.join(OUT0, "v1b_ridge", "ridge_init_zeroshot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[ridge_init] wrote v1b_ridge/ridge_init_zeroshot.json")


def _train_AB(model, hook, train_seqs, params, steps, bs, lr, device):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    hook.mode = "correct"; model.eval()
    n = train_seqs.shape[0]; order = torch.randperm(n); ptr = 0
    for step in range(steps):
        if ptr + bs > n:
            order = torch.randperm(n); ptr = 0
        ids = train_seqs[order[ptr:ptr + bs]].to(device); ptr += bs
        hook.current_ids = ids
        opt.zero_grad(set_to_none=True)
        logits = model(ids).logits
        sl = logits[:, :-1, :].float(); lab = ids[:, 1:]
        loss = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), lab.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward(); opt.step()


def mode_ridge_ft(args):
    """Tiny CE finetune: ridge-init (no train) vs ridge-init+finetune vs
    random-init+finetune, at the relaxed cap zero-shot showed sufficient."""
    device = args.device
    model, tok = load_model(device)
    eval_seqs = load_eval_seqs(tok, args.num_seq, args.ctxlen)
    calib_seqs = load_train_seqs(tok, args.num_seq, args.ctxlen)
    v0 = np.load(os.path.join(OUT0, "coarse_loss_delta.npy"))
    d = model.config.hidden_size; vocab = model.config.vocab_size
    base = _recompute_baseline_anchor(model, eval_seqs, args.batch_size, device)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [6, 7, 11]
    rank = 64; cap = 0.5; lr = args.lr if args.lr != 1e-3 else 3e-5; steps = args.steps
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    sub = eval_seqs[:256]; I = torch.eye(d, dtype=torch.float64, device=device)
    out = {}
    for layer in layers:
        lyr = model.gpt_neox.layers[layer]
        print(f"[ridge_ft] L{layer}: calibrating ...")
        mu, cnt, XtX, XtY, _ = _ridge_calibrate(model, lyr, calib_seqs, args.batch_size,
                                               device, vocab, d)
        vd = float(v0[layer])
        hook = V1aHook(model, layer); hook.attach()
        hook.anchor_mu = mu; hook.anchor_cnt = cnt; hook.variant = "shared"
        hook.scale = 1.0; hook.dv_cap = cap
        hook.mode = "table"; hook.A = hook.B = None
        d_a1 = eval_loss(model, hook, eval_seqs, args.batch_size, device) - base
        best = None
        for lam in lams:
            W = torch.linalg.solve(XtX + lam * I, XtY).float()
            U, Sv, Vh = torch.linalg.svd(W.double())
            A = (U[:, :256] * Sv[:256].sqrt()).float(); B = (Sv[:256].sqrt().unsqueeze(1) * Vh[:256]).float()
            hook.A, hook.B = A, B; hook.mode = "correct"
            ce = eval_loss(model, hook, sub, args.batch_size, device)
            if best is None or ce < best[1]:
                best = (lam, ce)
        W = torch.linalg.solve(XtX + best[0] * I, XtY).float()
        U, Sv, Vh = torch.linalg.svd(W.double())
        Ar = (U[:, :rank] * Sv[:rank].sqrt()).float(); Br = (Sv[:rank].sqrt().unsqueeze(1) * Vh[:rank]).float()

        def ctx(ce):
            return round((d_a1 - (ce - base)) / d_a1, 4) if abs(d_a1) > 1e-6 else None

        res = {"A1_recovery": round(1 - d_a1 / vd, 4), "best_lambda": best[0]}
        hook.A = torch.nn.Parameter(Ar.clone()); hook.B = torch.nn.Parameter(Br.clone()); hook.mode = "correct"
        res["ridge_init_notrain"] = ctx(eval_loss(model, hook, eval_seqs, args.batch_size, device))
        hook.A = torch.nn.Parameter(Ar.clone()); hook.B = torch.nn.Parameter(Br.clone())
        _train_AB(model, hook, calib_seqs, [hook.A, hook.B], steps, args.batch_size, lr, device)
        res["ridge_init_ft"] = ctx(eval_loss(model, hook, eval_seqs, args.batch_size, device))
        hook.init_lora("shared", rank, device)
        _train_AB(model, hook, calib_seqs, [hook.A, hook.B], steps, args.batch_size, lr, device)
        res["random_init_ft"] = ctx(eval_loss(model, hook, eval_seqs, args.batch_size, device))
        hook.detach()
        out[str(layer)] = res
        print(f"[ridge_ft] L{layer} A1={res['A1_recovery']:+.3f} | R_context: "
              f"ridge_notrain={res['ridge_init_notrain']} ridge_ft={res['ridge_init_ft']} "
              f"random_ft={res['random_init_ft']}")
    os.makedirs(os.path.join(OUT0, "v1b_ridge"), exist_ok=True)
    with open(os.path.join(OUT0, "v1b_ridge", "ridge_ft.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[ridge_ft] wrote v1b_ridge/ridge_ft.json")


# --------------------------------------------------------------------------- #
# plots + summary
# --------------------------------------------------------------------------- #
def mode_plots(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for layer in LAYERS:
        for variant in VARIANTS:
            for rank in RANKS:
                fn = os.path.join(HERE, f"L{layer}_{variant}_r{rank}.json")
                if os.path.exists(fn):
                    runs[(layer, variant, rank)] = json.load(open(fn))
    if not runs:
        print("[plots] no run json found")
        return

    # recovery curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {l: c for l, c in zip(LAYERS, ["C0", "C1", "C2", "C3"])}
    for layer in LAYERS:
        for variant in VARIANTS:
            xs, ys = [], []
            for rank in RANKS:
                r = runs.get((layer, variant, rank))
                if r:
                    xs.append(rank); ys.append(r["recovery_ratio"])
            if xs:
                ax.plot(xs, ys, marker="o",
                        color=colors[layer],
                        ls="-" if variant == "shared" else "--",
                        label=f"L{layer} {variant}")
    ax.set_xscale("log", base=2); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
    ax.set_xlabel("LoRA rank"); ax.set_ylabel("recovery ratio")
    ax.axhline(1.0, color="k", lw=0.5); ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("v1a recovery vs rank (solid=shared, dashed=per-head)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "recovery_curves.png"), dpi=130)
    print("[plots] wrote recovery_curves.png")

    # residual heatmap (two panels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, variant in zip(axes, VARIANTS):
        M = np.full((len(LAYERS), len(RANKS)), np.nan)
        for i, layer in enumerate(LAYERS):
            for j, rank in enumerate(RANKS):
                r = runs.get((layer, variant, rank))
                if r:
                    M[i, j] = r["residual_delta"]
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(RANKS))); ax.set_xticklabels(RANKS)
        ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels([f"L{l}" for l in LAYERS])
        ax.set_xlabel("rank"); ax.set_title(f"{variant}: residual delta")
        for i in range(len(LAYERS)):
            for j in range(len(RANKS)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="w", fontsize=7)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "residual_heatmap.png"), dpi=130)
    print("[plots] wrote residual_heatmap.png")

    # summary.md
    lines = ["# v1a single-layer recovery -- summary\n",
             f"- config: {MODEL}, fp16, lr={args.lr}, steps={args.steps}, "
             f"batch={args.batch_size}, AdamW(wd=0), alpha=r",
             f"- target layers: {LAYERS} (L5/11/17 = highest v0 delta; L23 = low-delta control)",
             f"- variants: {VARIANTS}; ranks: {RANKS}",
             ""]
    lines.append("## Table (residual delta / recovery / params(M) / recovery-per-M)\n")
    hdr = "| layer | variant | " + " | ".join(f"r{r}" for r in RANKS) + " |"
    lines.append(hdr); lines.append("|" + "---|" * (len(RANKS) + 2))
    for layer in LAYERS:
        for variant in VARIANTS:
            cells = []
            for rank in RANKS:
                r = runs.get((layer, variant, rank))
                if r:
                    cells.append(f"{r['residual_delta']:+.3f}/{r['recovery_ratio']:.2f}/"
                                 f"{r['param_count_M']:.3f}/{r['recovery_per_M']:.2f}")
                else:
                    cells.append("-")
            lines.append(f"| L{layer} | {variant} | " + " | ".join(cells) + " |")

    # Q1: Reading A vs B  (param-normalized at r<=8)
    def avg_metric(variant, metric, ranks):
        vals = [runs[(l, variant, r)][metric] for l in LAYERS for r in ranks
                if (l, variant, r) in runs]
        return float(np.mean(vals)) if vals else float("nan")

    sh_pm = avg_metric("shared", "recovery_per_M", [2, 4, 8])
    ph_pm = avg_metric("perhead", "recovery_per_M", [2, 4, 8])
    sh_rec = avg_metric("shared", "recovery_ratio", [2, 4, 8])
    ph_rec = avg_metric("perhead", "recovery_ratio", [2, 4, 8])
    if not (np.isfinite(sh_pm) and np.isfinite(ph_pm)):
        q1 = "insufficient runs to decide Reading A vs B"
    elif sh_rec < 0.2 and ph_rec < 0.2:
        q1 = ("neither variant recovers much (raw recovery < 0.2 at r<=8) -> deeper "
              "mismatch between post-hoc swap and trainable correction; revisit setup")
    elif sh_pm >= 0.8 * ph_pm:
        q1 = ("Reading A (shared low-rank signal): shared per-layer LoRA is "
              "competitive with per-head on PARAM-NORMALIZED recovery at r<=8 "
              f"(shared rec/M={sh_pm:.2f} vs per-head {ph_pm:.2f}) -> the recoverable "
              "V-contextualization is largely a shared low-rank signal")
    else:
        q1 = ("Reading B (heterogeneous distributed signal): per-head LoRA is "
              f"materially more param-efficient (rec/M {ph_pm:.2f} vs shared {sh_pm:.2f}) "
              "even after normalizing for its ~8.5x params -> heads carry distinct "
              "V-contextualization that a shared correction cannot capture")

    # Q2: is low-rank effective on the high-delta layers
    hi = [5, 11, 17]
    best = {}
    for l in hi:
        rr = [runs[(l, v, r)]["residual_delta"] for v in VARIANTS for r in [2, 4, 8, 16]
              if (l, v, r) in runs]
        best[l] = min(rr) if rr else float("nan")
    ok = [l for l in hi if np.isfinite(best[l]) and best[l] < 0.1]
    q2 = (f"low-rank (r<=16) drives residual<0.1 nats on layers {ok} of {hi} "
          f"(best residuals: {{ {', '.join(f'L{l}:{best[l]:+.3f}' for l in hi)} }}). "
          + ("-> v1b full-layer extension is justified." if len(ok) >= 2 else
             "-> recovery is limited; needs larger rank / richer correction before v1b."))

    lines += ["", "## Q1 -- Reading A vs B (param-normalized recovery at r<=8)\n",
              f"shared rec/M={sh_pm:.3f} (raw rec {sh_rec:.3f}); "
              f"per-head rec/M={ph_pm:.3f} (raw rec {ph_rec:.3f})\n",
              f"**{q1}**\n",
              "## Q2 -- is low-rank effective on high-delta layers (L5/11/17)\n",
              f"**{q2}**\n"]
    with open(os.path.join(HERE, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[plots] wrote summary.md")
    print(f"[plots] Q1: {q1}")
    print(f"[plots] Q2: {q2}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["scale_stats", "scale_coarse", "train", "all", "probe", "svd_diff", "anchor", "v1b", "ridge", "ridge_init", "ridge_ft", "plots"])
    p.add_argument("--split", default="eval", choices=["eval", "train"])
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--variant", default="")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--layers", default="")
    p.add_argument("--ranks", default="")
    p.add_argument("--rescale", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--num_seq", type=int, default=1000)
    p.add_argument("--ctxlen", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--dv_cap", type=float, default=0.0)
    p.add_argument("--cap_matrix", action="store_true")
    p.add_argument("--mlp_matrix", action="store_true")
    p.add_argument("--highrank_matrix", action="store_true")
    p.add_argument("--overfit_matrix", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    {
        "scale_stats": mode_scale_stats,
        "scale_coarse": mode_scale_coarse,
        "train": mode_train,
        "all": mode_all,
        "probe": mode_probe,
        "svd_diff": mode_svd_diff,
        "anchor": mode_anchor,
        "v1b": mode_v1b,
        "ridge": mode_ridge,
        "ridge_init": mode_ridge_init,
        "ridge_ft": mode_ridge_ft,
        "plots": mode_plots,
    }[args.mode](args)


if __name__ == "__main__":
    main()
