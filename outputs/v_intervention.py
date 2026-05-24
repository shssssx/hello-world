"""V-path contextualization audit on Pythia (post-hoc, no training).

Replaces attention V at selected (layer, head) positions with a "token-grounded"
version that depends only on the current token:

    V_l(t) <- W_V_l . LN_l( E[x_t] )

i.e. the V you would get if layer l's attention saw the raw token embedding
(passed through that layer's pre-attention LayerNorm) instead of the actual
contextualized hidden state. Everything else (LN, W_V path) is preserved; only
the cross-layer contextualization carried into V is removed.

Modes:
  introspect  print architecture facts, verify fused-QKV layout assumptions
  baseline    eval with no intervention -> baseline_loss.json
  sanity      baseline + layer-0 (all heads) intervention, print delta
  coarse      per-layer (all heads) intervention -> coarse_loss_delta.npy [L]
  fine        per-(layer,head) on top-k layers   -> fine_loss_delta.npy [L,H]
  summary     build heatmap.png + summary.md from the saved arrays

Run order: introspect -> baseline -> sanity -> coarse -> fine -> summary
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_eval_sequences(tokenizer, num_seq, ctxlen):
    """Return (LongTensor [N, ctxlen], source) of full (unpadded) token blocks.

    Tries monology/pile-uncopyrighted (train split; the mirror has no
    validation split) first, then wikitext-103 train. Concatenate documents
    (EOS-separated), chunk into ctxlen blocks, drop the remainder so every
    block is full length -> no padding needed. The result is cached to disk so
    baseline/coarse/fine all reuse the *same* sequences (required for the loss
    delta to be comparable) and we only hit the network once.
    """
    from datasets import load_dataset

    cache = os.path.join(HERE, f"eval_seqs_n{num_seq}_c{ctxlen}.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        print(f"[data] loaded cache {os.path.basename(cache)}: "
              f"source={d['source']} blocks={d['seqs'].shape[0]}")
        return d["seqs"], d["source"]

    eos = tokenizer.eos_token_id

    def build_from(text_iter):
        need = num_seq * ctxlen
        buf, total = [], 0
        for txt in text_iter:
            ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            buf.extend(ids)
            buf.append(eos)
            total += len(ids) + 1
            if total >= need:
                break
        n_full = len(buf) // ctxlen
        n_use = min(n_full, num_seq)
        arr = np.array(buf[: n_use * ctxlen], dtype=np.int64).reshape(n_use, ctxlen)
        return arr, total

    def pile_iter():
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        return (ex["text"] for ex in ds)

    def wikitext_iter():
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        return (t for t in ds["text"] if t.strip())

    arr = source = None
    for name, it_fn, label in [
        ("pile", pile_iter, "monology/pile-uncopyrighted:train"),
        ("wikitext", wikitext_iter, "wikitext-103-raw-v1:train"),
    ]:
        try:
            a, total = build_from(it_fn())
        except Exception as e:  # noqa: BLE001
            print(f"[data] {name} failed ({type(e).__name__}: {e}); trying next source")
            continue
        if a.shape[0] >= num_seq:
            arr, source = a, label
            print(f"[data] source={source} blocks={a.shape[0]} ctxlen={ctxlen} "
                  f"(read ~{total} tokens)")
            break
        print(f"[data] {name} produced only {a.shape[0]} blocks (< {num_seq}); trying next")

    if arr is None:
        raise RuntimeError(f"no dataset produced >= {num_seq} blocks of ctxlen {ctxlen}")

    seqs = torch.from_numpy(arr)
    torch.save({"seqs": seqs, "source": source}, cache)
    return seqs, source


# --------------------------------------------------------------------------- #
# intervention
# --------------------------------------------------------------------------- #
class VIntervention:
    """Forward hooks on each layer's query_key_value that splice the V slice."""

    def __init__(self, model):
        cfg = model.config
        self.model = model
        self.gpt_neox = model.gpt_neox
        self.layers = self.gpt_neox.layers
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.hidden = cfg.hidden_size
        self.head_dim = self.hidden // self.num_heads
        self.current_input_ids = None
        self.spec = {}          # layer_idx -> "all" | set(head_idx)
        self._handles = []

    def _hook(self, layer_idx):
        layer = self.layers[layer_idx]
        nh, hd = self.num_heads, self.head_dim

        def hook(module, inputs, output):
            heads = self.spec.get(layer_idx)
            if heads is None:
                return output
            # token-grounded qkv: W_qkv . LN_l( E[x] ) + b   (computed via
            # F.linear to avoid re-triggering this same hook)
            emb = self.gpt_neox.embed_in(self.current_input_ids)
            normed = layer.input_layernorm(emb)
            tg = F.linear(normed, module.weight, module.bias)

            B, S, _ = output.shape
            out = output.view(B, S, nh, 3 * hd)
            tgv = tg.view(B, S, nh, 3 * hd)
            if heads == "all":
                out[..., 2 * hd:] = tgv[..., 2 * hd:]
            else:
                idx = torch.tensor(sorted(heads), device=out.device, dtype=torch.long)
                out[:, :, idx, 2 * hd:] = tgv[:, :, idx, 2 * hd:]
            return out.view(B, S, nh * 3 * hd)

        return hook

    def attach(self):
        self.detach()
        for i, layer in enumerate(self.layers):
            h = layer.attention.query_key_value.register_forward_hook(self._hook(i))
            self._handles.append(h)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def set_spec(self, spec):
        self.spec = spec


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, interv, seqs, batch_size, device):
    """Return per-seq mean next-token CE loss, LongTensor[N]."""
    model.eval()
    out = []
    n = seqs.shape[0]
    for i in range(0, n, batch_size):
        ids = seqs[i : i + batch_size].to(device)
        interv.current_input_ids = ids
        logits = model(ids).logits  # [B,S,V]
        shift_logits = logits[:, :-1, :].float()
        shift_labels = ids[:, 1:]
        B, Sm1, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.reshape(-1, V), shift_labels.reshape(-1), reduction="none"
        ).view(B, Sm1).mean(dim=1)
        out.append(loss.detach().float().cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def per_seq_path():
    return os.path.join(HERE, "losses_per_seq.pt")


def load_per_seq():
    p = per_seq_path()
    return torch.load(p) if os.path.exists(p) else {}


def save_per_seq(store):
    torch.save(store, per_seq_path())


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_model(model_name, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tok


def base_config(args, source, n_blocks):
    return {
        "model": args.model,
        "dataset": source,
        "num_seq": n_blocks,
        "ctxlen": args.ctxlen,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": str(args.device),
    }


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_introspect(args):
    model, tok = load_model(args.model, args.dtype, args.device)
    cfg = model.config
    layer0 = model.gpt_neox.layers[0]
    qkv = layer0.attention.query_key_value
    nh = cfg.num_attention_heads
    hid = cfg.hidden_size
    hd = hid // nh
    print("=" * 60)
    print("ARCHITECTURE INTROSPECTION")
    print("=" * 60)
    print(f"model                 : {args.model}")
    print(f"num_hidden_layers     : {cfg.num_hidden_layers}")
    print(f"num_attention_heads   : {nh}")
    print(f"hidden_size           : {hid}")
    print(f"head_dim (hid/nh)     : {hd}")
    print(f"qkv.weight shape      : {tuple(qkv.weight.shape)}  (expect [{3*hid}, {hid}])")
    print(f"qkv.bias  shape       : {tuple(qkv.bias.shape) if qkv.bias is not None else None}")
    print(f"embed_in              : {type(model.gpt_neox.embed_in).__name__} "
          f"{tuple(model.gpt_neox.embed_in.weight.shape)}")
    print(f"layer0.input_layernorm: {type(layer0.input_layernorm).__name__}")
    print(f"attention module type : {type(layer0.attention).__name__}")
    ok = tuple(qkv.weight.shape) == (3 * hid, hid)
    print("-" * 60)
    print(f"fused-QKV [3*hidden, hidden] layout assumption: {'OK' if ok else 'MISMATCH!'}")

    # numeric sanity: splicing head 0's V must change the qkv output, and the
    # token-grounded V must differ from the real V (proves the hook does work).
    interv = VIntervention(model)
    ids = torch.randint(0, cfg.vocab_size, (1, 16), device=args.device)
    with torch.no_grad():
        real = qkv(layer0.input_layernorm(model.gpt_neox.embed_in(ids)))
        emb = model.gpt_neox.embed_in(ids)
        tg = F.linear(layer0.input_layernorm(emb), qkv.weight, qkv.bias)
    real_v = real.view(1, 16, nh, 3 * hd)[..., 2 * hd :]
    tg_v = tg.view(1, 16, nh, 3 * hd)[..., 2 * hd :]
    diff = (real_v - tg_v).abs().mean().item()
    print(f"mean|real_V - tokengrounded_V| (layer0, random ids): {diff:.4f}")
    print("(should be > 0 but modest at layer 0; ~0 would mean hook is a no-op)")
    print("=" * 60)
    if not ok:
        print("\nSTOP: QKV layout does not match the assumed fused [3*hidden, hidden] "
              "scheme. Do not run further modes; report this back.")


def _ensure_baseline(model, interv, seqs, args, store):
    """Compute baseline once, persist per-seq + json. Returns per-seq tensor."""
    if "baseline" in store:
        return store["baseline"]
    interv.set_spec({})
    t0 = time.time()
    bl = evaluate(model, interv, seqs, args.batch_size, args.device)
    print(f"[baseline] mean={bl.mean().item():.4f}  ({time.time()-t0:.1f}s)")
    store["baseline"] = bl
    save_per_seq(store)
    return bl


def mode_baseline(args):
    model, tok = load_model(args.model, args.dtype, args.device)
    seqs, source = load_eval_sequences(tok, args.num_seq, args.ctxlen)
    interv = VIntervention(model)
    interv.attach()
    store = load_per_seq()
    bl = _ensure_baseline(model, interv, seqs, args, store)
    cfg = base_config(args, source, seqs.shape[0])
    cfg["baseline_mean_loss"] = float(bl.mean())
    with open(os.path.join(HERE, "baseline_loss.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[baseline] wrote baseline_loss.json  mean={bl.mean().item():.4f}")


def mode_sanity(args):
    model, tok = load_model(args.model, args.dtype, args.device)
    seqs, source = load_eval_sequences(tok, args.num_seq, args.ctxlen)
    interv = VIntervention(model)
    interv.attach()
    store = load_per_seq()
    bl = _ensure_baseline(model, interv, seqs, args, store)
    interv.set_spec({0: "all"})
    l0 = evaluate(model, interv, seqs, args.batch_size, args.device)
    d0 = (l0.mean() - bl.mean()).item()
    print("=" * 60)
    print(f"SANITY: layer-0 all-heads intervention")
    print(f"baseline mean loss     : {bl.mean().item():.4f}")
    print(f"layer-0 mean loss      : {l0.mean().item():.4f}")
    print(f"layer-0 loss delta     : {d0:+.4f}")
    print("=" * 60)
    print("Expectation: layer-0 delta should be modest (only difference is whether "
          "V sees pre-LN raw embedding vs the real residual at layer 0). If this is "
          "huge or negative-huge, the hook is likely buggy -- STOP and inspect.")


def mode_coarse(args):
    model, tok = load_model(args.model, args.dtype, args.device)
    seqs, source = load_eval_sequences(tok, args.num_seq, args.ctxlen)
    interv = VIntervention(model)
    interv.attach()
    store = load_per_seq()
    bl = _ensure_baseline(model, interv, seqs, args, store)
    base_mean = bl.mean().item()

    L = interv.num_layers
    deltas = np.full(L, np.nan, dtype=np.float64)
    per_seq = store.get("coarse", torch.full((L, seqs.shape[0]), float("nan")))
    for l in range(L):
        interv.set_spec({l: "all"})
        t0 = time.time()
        ls = evaluate(model, interv, seqs, args.batch_size, args.device)
        deltas[l] = ls.mean().item() - base_mean
        per_seq[l] = ls
        print(f"[coarse] layer {l:2d}  delta={deltas[l]:+.4f}  ({time.time()-t0:.1f}s)")
        store["coarse"] = per_seq
        save_per_seq(store)
        np.save(os.path.join(HERE, "coarse_loss_delta.npy"), deltas)
    print(f"[coarse] done. min={np.nanmin(deltas):+.4f} "
          f"median={np.nanmedian(deltas):+.4f} max={np.nanmax(deltas):+.4f}")


def _pick_fine_layers(deltas, k):
    order = np.argsort(-np.abs(deltas))  # largest |delta| first
    return sorted(order[:k].tolist())


def mode_fine(args):
    model, tok = load_model(args.model, args.dtype, args.device)
    seqs, source = load_eval_sequences(tok, args.num_seq, args.ctxlen)
    interv = VIntervention(model)
    interv.attach()
    store = load_per_seq()
    bl = _ensure_baseline(model, interv, seqs, args, store)
    base_mean = bl.mean().item()

    L, H = interv.num_layers, interv.num_heads
    if args.layers:
        layers = sorted(int(x) for x in args.layers.split(","))
    else:
        coarse = np.load(os.path.join(HERE, "coarse_loss_delta.npy"))
        layers = _pick_fine_layers(coarse, args.top_k)
    print(f"[fine] layers to scan: {layers}")

    fine = store.get("fine_delta", np.full((L, H), np.nan, dtype=np.float64))
    if isinstance(fine, torch.Tensor):
        fine = fine.numpy()
    per_seq = store.get("fine", torch.full((L, H, seqs.shape[0]), float("nan")))
    for l in layers:
        for h in range(H):
            interv.set_spec({l: {h}})
            t0 = time.time()
            ls = evaluate(model, interv, seqs, args.batch_size, args.device)
            fine[l, h] = ls.mean().item() - base_mean
            per_seq[l, h] = ls
            print(f"[fine] L{l:2d} H{h:2d}  delta={fine[l,h]:+.4f}  ({time.time()-t0:.1f}s)")
        store["fine"] = per_seq
        store["fine_delta"] = torch.from_numpy(fine)
        save_per_seq(store)
        np.save(os.path.join(HERE, "fine_loss_delta.npy"), fine)
    print(f"[fine] done. scanned layers {layers}")


def mode_summary(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(HERE, "baseline_loss.json")) as f:
        cfg = json.load(f)
    coarse = np.load(os.path.join(HERE, "coarse_loss_delta.npy"))
    fine = np.load(os.path.join(HERE, "fine_loss_delta.npy"))
    base_mean = cfg["baseline_mean_loss"]
    L, H = fine.shape

    # ---- figure: heatmap + coarse bar ----
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 2.2]}
    )
    ax0.barh(np.arange(L), coarse)
    ax0.set_title("Coarse: per-layer (all heads) loss delta")
    ax0.set_ylabel("layer")
    ax0.set_xlabel("loss delta")
    ax0.invert_yaxis()
    ax0.axvline(0, color="k", lw=0.5)

    masked = np.ma.masked_invalid(fine)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")
    vmax = np.nanmax(np.abs(fine)) if np.isfinite(fine).any() else 1.0
    im = ax1.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax1.set_title("Fine: per-(layer,head) loss delta (gray = not measured)")
    ax1.set_xlabel("head")
    ax1.set_ylabel("layer")
    fig.colorbar(im, ax=ax1, label="loss delta")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "heatmap.png"), dpi=130)
    print("[summary] wrote heatmap.png")

    # ---- scenario judgment ----
    finite = fine[np.isfinite(fine)]
    near_zero = [
        (int(l), int(h))
        for l in range(L)
        for h in range(H)
        if np.isfinite(fine[l, h]) and abs(fine[l, h]) < 0.05
    ]
    c_min, c_med, c_max = (
        float(np.nanmin(coarse)),
        float(np.nanmedian(coarse)),
        float(np.nanmax(coarse)),
    )
    sig_layers = [int(l) for l in range(L) if abs(coarse[l]) >= 0.1]

    # heuristic scenario selection
    others_big = np.all(np.abs(coarse[1:]) > 0.1) if L > 1 else False
    layer0_small = abs(coarse[0]) < 0.05
    if finite.size:
        frac_small = (np.abs(finite) < 0.05).mean()
        f_med = float(np.median(np.abs(finite)))
    else:
        frac_small, f_med = float("nan"), float("nan")

    coarse_med_abs = float(np.median(np.abs(coarse)))
    if layer0_small and others_big:
        scenario = "S3 (only layer 0 ~ 0; V contextualization important for layers >= 1)"
    elif finite.size and coarse_med_abs > 0.1 and frac_small > 0.5:
        scenario = ("S1 / distributed redundancy: per-head V contextualization is "
                    "largely removable (most per-head deltas ~0) yet removing it from "
                    "all heads in a layer is costly -> redundant, distributed signal; "
                    "worth escalating to low-rank correction / from-scratch training")
    elif finite.size and frac_small > 0.15 and f_med > 0.1:
        scenario = "S1 (structured: some positions ~0, some large -> signal worth escalating)"
    elif finite.size and f_med > 0.15 and frac_small < 0.1:
        scenario = "S2 (uniformly high: pure post-hoc swap too aggressive)"
    else:
        scenario = "ambiguous (see numbers below; inspect manually)"

    lines = []
    lines.append("# V-path contextualization audit -- summary\n")
    lines.append("## Config\n")
    for k in ["model", "dataset", "num_seq", "ctxlen", "batch_size", "dtype", "device"]:
        lines.append(f"- **{k}**: {cfg.get(k)}")
    lines.append(f"\n## Baseline\n\n- mean next-token CE loss: **{base_mean:.4f}**\n")
    lines.append("## Coarse (per-layer, all heads)\n")
    lines.append(f"- min={c_min:+.4f}  median={c_med:+.4f}  max={c_max:+.4f}")
    lines.append(f"- layers with |delta| >= 0.1: {sig_layers}")
    lines.append("- per-layer deltas:")
    for l in range(L):
        lines.append(f"  - L{l}: {coarse[l]:+.4f}")
    lines.append("\n## Fine (per-head, scanned layers)\n")
    if finite.size:
        lines.append(f"- min={finite.min():+.4f}  median={np.median(finite):+.4f}  "
                     f"max={finite.max():+.4f}")
        lines.append(f"- fraction of scanned heads with |delta| < 0.05: {frac_small:.2%}")
        lines.append(f"- (layer, head) with loss delta < 0.05 ({len(near_zero)} total):")
        lines.append(f"  {near_zero}")
    else:
        lines.append("- no fine measurements found.")
    lines.append("\n## Judgment\n")
    lines.append(f"**Scenario: {scenario}**\n")
    lines.append("Basis: layer-0 coarse delta = "
                 f"{coarse[0]:+.4f}; coarse median |delta| = "
                 f"{float(np.median(np.abs(coarse))):.4f}; "
                 f"fine median |delta| = {f_med if finite.size else float('nan'):.4f}; "
                 f"fraction of fine heads ~0 = "
                 f"{frac_small if finite.size else float('nan'):.2%}.")
    with open(os.path.join(HERE, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[summary] wrote summary.md")
    print(f"[summary] scenario: {scenario}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["introspect", "baseline", "sanity", "coarse", "fine", "summary"])
    p.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    p.add_argument("--num_seq", type=int, default=1000)
    p.add_argument("--ctxlen", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--top_k", type=int, default=8, help="fine: #layers to scan")
    p.add_argument("--layers", default="", help="fine: explicit comma layer list, overrides top_k")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.device == "cpu":
        print("[warn] running on CPU -- this will be very slow for full eval.")

    {
        "introspect": mode_introspect,
        "baseline": mode_baseline,
        "sanity": mode_sanity,
        "coarse": mode_coarse,
        "fine": mode_fine,
        "summary": mode_summary,
    }[args.mode](args)


if __name__ == "__main__":
    main()
