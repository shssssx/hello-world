# V-path contextualization audit -- summary

## Config

- **model**: EleutherAI/pythia-410m-deduped
- **dataset**: monology/pile-uncopyrighted:train
- **num_seq**: 1000
- **ctxlen**: 1024
- **batch_size**: 16
- **dtype**: fp16
- **device**: cuda

## Baseline

- mean next-token CE loss: **2.1178**

## Coarse (per-layer, all heads)

- min=+0.0000  median=+0.3225  max=+0.6230
- layers with |delta| >= 0.1: [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
- per-layer deltas:
  - L0: +0.0000
  - L1: +0.3209
  - L2: +0.0878
  - L3: +0.3426
  - L4: +0.1255
  - L5: +0.6230
  - L6: +0.3236
  - L7: +0.2651
  - L8: +0.3667
  - L9: +0.2248
  - L10: +0.3811
  - L11: +0.4536
  - L12: +0.4071
  - L13: +0.3866
  - L14: +0.3348
  - L15: +0.2702
  - L16: +0.4191
  - L17: +0.5728
  - L18: +0.3215
  - L19: +0.3774
  - L20: +0.1679
  - L21: +0.1740
  - L22: +0.0942
  - L23: +0.0775

## Fine (per-head, scanned layers)

- min=+0.0009  median=+0.0120  max=+0.0679
- fraction of scanned heads with |delta| < 0.05: 97.66%
- (layer, head) with loss delta < 0.05 (125 total):
  [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (5, 0), (5, 1), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (14, 0), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (14, 15), (17, 0), (17, 1), (17, 2), (17, 3), (17, 4), (17, 5), (17, 6), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (20, 0), (20, 1), (20, 2), (20, 3), (20, 4), (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (20, 10), (20, 11), (20, 12), (20, 13), (20, 14), (20, 15), (23, 0), (23, 1), (23, 2), (23, 3), (23, 4), (23, 5), (23, 6), (23, 7), (23, 8), (23, 9), (23, 10), (23, 11), (23, 12), (23, 13), (23, 14), (23, 15)]

## Judgment

**Scenario: S1 / distributed redundancy: per-head V contextualization is largely removable (most per-head deltas ~0) yet removing it from all heads in a layer is costly -> redundant, distributed signal; worth escalating to low-rank correction / from-scratch training**

Basis: layer-0 coarse delta = +0.0000; coarse median |delta| = 0.3225; fine median |delta| = 0.0120; fraction of fine heads ~0 = 97.66%.
