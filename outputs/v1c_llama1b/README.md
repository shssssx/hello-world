# v1c_llama1b - Llama-3.2-1B cross-architecture replication

Mirrors the Pythia v1a / v1b value-path audit at minimum viable scope:
A0 / A1 anchors + closed-form ridge. No SGD-trained variants here -
the Pythia results already localize the bottleneck to optimization,
and the Llama replication answers a different question (does the
token/context decomposition generalize across architecture, not
"can SGD find ridge").

## Status

| component                       | state                                       |
|---------------------------------|---------------------------------------------|
| Hook (`VHookLlama`)             | written, **not yet exercised**              |
| A1 calibration                  | written, not yet run                        |
| Ridge calibration               | written, not yet run                        |
| `mode_audit` (depth profile)    | written, not yet run                        |
| `mode_ridge` (R_context probe)  | written, not yet run                        |
| Weights download                | NOT triggered (gated repo, needs HF login)  |
| Calib seqs                      | will be cached at `outputs/llama_calib_seqs_n4000_c1024.pt` on first run |

Per current plan in `outputs/notes/paper_diffs.md`: Llama runs are
deferred until seq=50 sgd_pressure outcome locks the §5.5 narrative,
to avoid re-running Llama if cap-fix-driven framing changes propagate.

## Architecture differences vs Pythia-410M (GPT-NeoX)

- **RMSNorm** with learned weight at the decoder-layer level
  (`model.model.layers[L].input_layernorm`), applied BEFORE
  `self_attn`. By the time `v_proj` runs, hidden state is already
  RMS-normed -- so post-norm input to `v_proj` is the right `X` for
  the ridge fit, no extra normalization step needed.
- **Grouped-query attention** with 32 query heads, 8 KV heads,
  group size 4, `head_dim=64`. `v_proj` output is
  `num_kv_heads * head_dim = 512`, NOT `d_model=2048`. All anchor
  tables, ridge maps, and (would-be) LoRA factors live in the
  512-dim V space; storage and compute scale accordingly (smaller
  than Pythia per-layer V).
- **Separate q_proj / k_proj / v_proj**, not fused. Hook on
  `v_proj` directly -- no QKV slice surgery, no per-head reshape
  of a fused tensor.
- **Rotary on Q,K only**, as in Pythia. V is position-free, so the
  layer-0 V sanity (V equals v_proj of LN of raw embedding) still
  holds.
- `d_model=2048`, 16 layers, vocab=128256.

## How to run (when unblocked)

Requires HF login (`huggingface-cli login`) for the gated
`meta-llama/Llama-3.2-1B` repo.

```bash
# from outputs/v1c_llama1b/
python v1c_llama_audit.py --mode audit --layers 0,3,5,7,9,11,13,15
# -> writes audit.json: per-layer Delta_A0, Delta_A1, A1_recovery

python v1c_llama_audit.py --mode ridge --layers 5,7,9
# -> writes ridge.json: per-layer R_context at rank=64, cap=0.5

# both default to 4000 calib seqs / 1024 ctxlen / batch_size 4.
```

Approximate runtimes on a 4090 at fp16, batch 4, 1024 ctx:
- `mode audit`, 8 layers: ~30 min (one calib pass + two eval passes per layer)
- `mode ridge`, 3 layers: ~25 min (calib + XtX/XtY + lambda sweep + eval)

## Hooks and modes

`VHookLlama` is one hook on `model.model.layers[L].self_attn.v_proj`:
- pre-forward captures `x` = post-RMSNorm input (the ridge's `X`)
- post-forward rewrites `output` (= `V_real`) according to `self.mode`:
  - `off`: passthrough (sanity baseline)
  - `collect`: passthrough but accumulate per-token-id mean V_real
    into `anchor_mu`, `anchor_cnt` (used by `calibrate_a1`)
  - `anchor`: `V = anchor_mu[x_t]` (A0 if `anchor_mu` is the fallback
    table; A1 if calibrated)
  - `correct`: `V = anchor_mu[x_t] + (x @ A) @ B`, with the
    differentiable global-scalar soft cap (same form as the post-fix
    `v1a_correction.py` cap)

## Output paths

- `outputs/v1c_llama1b/audit.json` -- per-layer A0/A1 recovery profile
- `outputs/v1c_llama1b/ridge.json` -- selected-layer R_context
- `outputs/llama_calib_seqs_n4000_c1024.pt` -- tokenized calib cache (shared)

## Not implemented (deliberate scope cap)

- SGD-trained LoRA / MLP variants
- Per-head ablation grid
- Multi-rank ridge sweep (only rank=64 here)
- Cross-scale 160M-style replication (already covered in `v1b_160m/`)
- Per-token cap (uses global scalar to match v1a setup)

If the Pythia §5.5 SGD pressure narrative needs cross-architecture
support (i.e., reviewer asks "does SGD also fail on Llama?"), add an
`sgd_pressure` mode here at that point; the hook infrastructure is
already in place.
