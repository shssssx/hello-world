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

    def attach(self):
        self.detach()
        self._handle = self.qkv.register_forward_hook(self._hook)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _token_table_V(self):
        """[B,S,nh,hd] token-table V = W_V . LN_l(E[x]); optionally rescaled."""
        emb = self.gpt_neox.embed_in(self.current_ids)
        normed = self.layer.input_layernorm(emb)
        tg = F.linear(normed, self.qkv.weight, self.qkv.bias)      # [B,S,3d]
        B, S, _ = tg.shape
        tgv = tg.view(B, S, self.nh, 3 * self.hd)[..., 2 * self.hd:]   # [B,S,nh,hd]
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
            v = v + self._correction(inputs[0]).to(v.dtype)
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
        if variant == "shared":
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
        "rescale": bool(use_rescale), "train_steps": args.steps,
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
                   choices=["scale_stats", "scale_coarse", "train", "all", "plots"])
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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    {
        "scale_stats": mode_scale_stats,
        "scale_coarse": mode_scale_coarse,
        "train": mode_train,
        "all": mode_all,
        "plots": mode_plots,
    }[args.mode](args)


if __name__ == "__main__":
    main()
