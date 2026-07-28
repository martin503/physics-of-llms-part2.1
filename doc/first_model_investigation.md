# First model investigation: closing the Figure-3 gaps (eq15 and eq20)

The first trained model — the converged 100k checkpoint (`models/100k_model/gpt-rope-igsm-fixed`,
GPT-2 + RoPE on iGSM-med) — nails the easy slice but has **two** Figure-3 (med) gaps:

```
Figure-3 (med) -- converged 100k model
  op_le15   0.985      (easy, ≤15 ops)
  op_eq15   0.360      (hard, exactly 15 ops -- in-distribution)
  op_eq20   0.000      (OOD, 20 ops)
  op_eq21-23 0.000
```

This document records how we closed both. **eq15 was easy** — a data problem, fixed in a few dozen
steps (§1). **eq20 was the deep one** — first diagnosed down to a compositional-planning limit (§2,
including the *"force it to reason longer"* experiment we reproduce and refute), then fixed by a
**training-regime** change (continue from a plastic checkpoint at a low enough LR), reaching **stable
eq20 ≈ 0.46–0.51** while holding le15 ≈ 0.98 / eq15 ≈ 0.97 (§3).

---

## 1. eq15 — the easy axis: uniform hard-data fixes it in a few steps

**Cause = hard-data scarcity, not capacity.** The iGSM generator samples the target op count as
`min(U, U)`, so op = 15 is ~29× rarer than op = 1; the model is simply starved of hard problems.

**Fix = balanced data + a tiny constant LR, fine-tuned from the converged 100k checkpoint.** On uniform
eq11–15 data (`data/igsm_raw_10M_eq11_15_packed768`), gbs64/ctx768, constant LR (no warmup, no
schedule), eq15 climbs from **0.32 → ~1.0 in ~50–100 steps (≈ 2.5–5 M tokens)**, and le15 stays
~0.98–0.99. Saturation step scales **∝ 1/LR** (halve the LR → ~2× the steps), all the way down to
5e-7:

| LR (constant, gbs64/ctx768) | eq15 ≥ 0.99 first reached | ≈ tokens | le15 (held) |
|---|---|---|---|
| 8e-6 | step ~50 (1.000 @ 100) | 2.5 M | 0.98 |
| 4e-6 | step ~100 | 5 M | 0.99 |
| 2e-6 | step ~150 | 7.4 M | 0.99 |
| 1e-6 | step ~300 | 15 M | 0.99 |
| 5e-7 | step ~400 | 20 M | 0.99 |

le15 is best at the *earliest* checkpoint and only erodes with more training — so "overfitting" here
just means training too long at too-high an LR; early-stop at saturation. **eq20 stays 0.0 at every
checkpoint** (the data is op ≤ 15) — that is a separate ceiling, investigated in §2. Drivers:
`scripts/lr_stair_{train,eval,report}.{sh,py}`.

---

## 2. eq20 — diagnosing the = 0 failure (preliminary checks)

### 2.1 `eval_loss ≈ 0.44` is an irreducible entropy floor, not a failure signal
`notebooks/eval_loss_floor_irreducible.ipynb` (cache `notebooks/_floor_cache.pt`).

The eval loss sits at ~0.44 and won't budge — is that undertraining / overfitting / precision? No: it is
the **irreducible entropy of fresh per-problem random draws**. Per-token NLL over 196k tokens (fp32
checkpoint): mean **0.442**, but **median per-token loss ≈ 0** (most tokens predicted perfectly).
Decomposition (sums to 100%): VAR-assignment 22.5% (mean 1.19 nats), NAME-first-mention 49.0% (1.04),
NAME-continuation 12.1% (0.15), STRUCTURE 16.4% (0.20). The random draws sit at their Bayes floor
(VAR assignments mean 3.56 nats vs `log(52)=3.95` — it even *beats* uniform by exploiting the
without-replacement constraint). The top 20% of tokens carry **99.2%** of the loss; an oracle that
removes random-draw tokens (8.6% of tokens, 58.6% of loss) leaves a residual of just **0.20 nats** on
the rest.

**Takeaway:** a *broken* bf16@100k run and a *working* fp32 run both sit at this same 0.44, so
`eval_loss` **cannot distinguish a working model from a broken one**. Use generation accuracy, never
loss.

### 2.2 bf16 vs fp32: precision is **not** the differentiator
`notebooks/bf16_vs_fp32_reasoning.ipynb` (examples `notebooks/bf16_vs_fp32_examples.json`).

An earlier single sample suggested bf16 "degenerates to copy-and-stop" while fp32 reasons. Side-by-side
greedy generation on the *same* Figure-3 problems (eq15/eq20/eq23), with every `N op M = R` clause
regex-verified mod 23, **corrects** that: bf16@100k reasons just like fp32 — long chain-of-thought
(~245–318 tokens), **every arithmetic step correct mod 23 in both**. The two are nearly
indistinguishable. **Precision is ruled out** as the cause of the OOD failure.

### 2.3 The failure mode: premature stop at op ≈ 15 (the "length ceiling")
`notebooks/eq20_failure_analysis.ipynb` (cache `notebooks/_eq20_diag.pkl`; the full per-sample
extract is ~140 MB and not committed). This is the authoritative quantification — it runs iGSM's strict `true_correct` (final answer **and**
every mod-23 calculation **and** every dependency edge) on the full eval log.

On `op_eq20` (requires 20 ops) the model produces **14 ops (54%) or 15 ops (45%) — never 20**. On
`op_eq15` it reaches the full 15 only ~36% of the time (= exactly its correct answers); the other
~63% stop at 14. Critically, **`wrong_cal = 0` in 100%** of these premature stops — the arithmetic on
the prefix the model *does* emit is flawless. Generations are ~286 tokens against a 2048 cap, so this
is **not truncation**: the model *chooses* to emit the answer sentinel `[224]` ~5–6 ops short of the
queried variable, then outputs the last intermediate it computed (wrong final answer).

**Takeaway:** the eq20 = 0 is a **length / planning ceiling at the training `max_op = 15`**, not
arithmetic, not precision, not truncation.

### 2.4 Force-continue: "it just learned the boundary" — tested and refuted
Reproduced on the converged 100k model (`gpt-rope-igsm-fixed`; the original fp32 target is no
longer on disk, but it exhibits the identical eq20 = 0 / stop-at-op≈15 behavior).

**Hypothesis:** maybe the model only *learned to stop* at op ≈ 15 — a reflex, not a capability limit.
If we **ban the answer token `[224]` and EOS** during generation so it literally cannot stop, perhaps it
will keep reasoning to op 20 and get the answer right.

**Method (`ForceThenAnswer` logits processor):** for the first `force` generated tokens, set `[224]`
and EOS to `-inf`; at token `force`, force the `[224]` answer transition; after that, free greedy so it
emits its answer number then EOS. Compare `force=0` (baseline greedy) vs `force=420`.

**Result (N=96 eq20 problems):**

| setting | eq20 accuracy |
|---|---|
| force = 0 (baseline) | **0 / 96 = 0.000** |
| force = 420 (forced past the stop) | **0 / 96 = 0.000** |

Forcing past the premature stop **does not raise eq20 at all**. The diagnostic on the forced
generations shows why — they are **arithmetically correct but structurally broken**:

```
eq20 #1  n_op=20  queried="Cincinnati Zoo's Insect House"  gold=18
  forced(420): 17 arith clauses, 0 wrong mod-23 | reached queried name? yes
  ... H = 18 + 22 = 17. so H = 20 + H = 20 + 17 = 14. so H = 20 + H = 20 + 14 = 11.
      so H = 11 + H = 11 + 11 = 22. Define ... Cockroach Corner as M; so M = 21 * K = 21 * 21 = 4. 4
```

Every individual `a op b = r` clause is correct mod 23 (**0 wrong**), but the *structure* is
incoherent — self-referential loops (`H = 20 + H`, impossible in a dependency DAG), variables recomputed
out of order, and the queried variable's name appears only in a corrupted context with a wrong value.
The forced `[224]` answer captures that wrong value.

**Conclusion — the boundary is not just a learned stop.** The model genuinely **cannot compose the
correct 20-step dependency chain**. Forced to reason longer, it produces locally-correct arithmetic
assembled into globally-wrong structure. The op ≈ 15 ceiling is a **compositional-planning limitation
at this training regime**, not a stop-token reflex — so "force it to reason longer" cannot work, and
the fix has to come from training, not decoding.

---

## 3. eq20 — the fix: training regime (not data, not architecture)

### 3.1 Plasticity: continue from the *less-converged* 80k checkpoint
Every fine-tune from the **converged 100k** model kept eq20 at exactly 0.000 (any LR, ctx, length).
The breakthrough was a different **starting point**: instead of SFT-ing the converged model, **continue
training from the less-converged 80k checkpoint** (`models/100k_model/checkpoint-80000-fixed`). At
lr2e-4, gbs256, ctx2048, warmup100, cosine floored at 7e-5 (= the original schedule's value at step 90k),
bf16 mixed precision, on balanced op1–15 data, this produced the **first non-zero eq20 ever** — but it
was **transient**:

| checkpoint | le15 | eq15 | eq20 |
|---|---|---|---|
| base 80k (untrained) | 0.977 | 0.000 | 0.000 |
| step 1000 | 0.977 | 0.984 | **0.119** |
| step 2000 | 0.992 | 1.000 | 0.000 |

eq20 = 0.119 at step 1000 (confirmed 61/512), then **crashes to 0 by step 2000** as the model over-fits
op ≤ 15 (eq15 0.984 → 1.0 *while* eq20 0.12 → 0). Drivers: `scripts/op80k_{train,eval,report}.{sh,py}`.
The crash is an **over-fitting artifact of too-high LR**, not a
wall — which §3.2 exploits.

### 3.2 Lower the LR to 5e-5: eq20 goes high **and** stable
Same plastic-80k recipe but **lr5e-5** (continue from `checkpoint-80000-fixed`, balanced op1–15
`data/igsm_raw_30M_eq1_15_packed768`, warmup100, cosine → ~0, ctx768, gbs512 = per_dev32×accum8,
**fa2 + pure-bf16**). eq20 across all 20 checkpoints (n=128 each):

| step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 | 1300–2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eq20 | 0.00 | 0.41 | 0.43 | 0.50 | **0.51** | 0.49 | 0.45 | **0.51** | 0.46 | 0.48 | 0.46 | 0.44 | ~0.46 (flat) |

- **Sharp onset:** 0.00 at step 100 (still in warmup) → 0.41 at step 200.
- **Peak 0.508 @ step 500/800** (65/128) — 4.3× the lr2e-4 runs' 0.119; best eq20 in the project.
- **No crash:** after ~step 900 it plateaus at **~0.46 and stays flat through step 2000** (steps 900→2000
  all ≈ 0.46). The in-dist/OOD over-fitting tradeoff that zeroed eq20 at lr2e-4 does **not** occur at
  lr5e-5.

| checkpoint | le15 | eq15 | eq20 |
|---|---|---|---|
| base 80k (untrained) | 0.977 | 0.000 | 0.000 |
| **step 500 (eq20 peak)** | **0.984** | **0.977** | **0.508** |
| step 2000 (final, plateau) | 0.992 | 0.969 | 0.461 |

Step-500 is the best all-round checkpoint produced: the first with **high eq20 *and* high le15/eq15
*and* stability**. (fa2 + pure-bf16 ran cleanly here — no loss floor / grad spikes / NaN, loss
0.46 → 0.43, grad_norm 0.02–0.05; the "pure-bf16 broken without QK-norm" concern is regime-dependent,
not categorical.)

**Bottom line:** eq20 is largely a **training-regime** problem, not a data problem. The same op ≤ 15
data + the same plastic 80k base reaches **stable ~0.46–0.51 eq20** just by using a low enough LR (5e-5)
that the model doesn't over-fit away the compositional length generalization — while keeping
le15 ≈ 0.98 and eq15 ≈ 0.97. op > 15 data is still likely needed to push toward the paper's ~1.0 on
**eq20–23**, but the route is now high *and* stable, not transient.

---

## Reproducibility

- **eq15 fine-tune (§1):** `scripts/lr_stair_{train,eval,report}.{sh,py}`.
- **eq20 fix (§3):** `scripts/op80k768_train.sh` with
  `LR=5e-5 MAX_STEPS=2000 SAVE_STEPS=100 MIN_LR_RATE=1e-5 OUT=disc_80k_op1_15_lr2e4_gbs512_ctx768_fa2_1k_2`
  (dir name says `lr2e4` but the run is lr5e-5). ~3 s/step at gbs512/ctx768/fa2 → ~1.7 h for 2k.
- **Training / eval scripts:** `scripts/` — `lr_stair_{train,eval,report}.{sh,py}` (§1 eq15),
  `op80k_{train,eval,report}.{sh,py}` (§3.1 plasticity), `op80k768_train.sh` (§3.2 lr5e-5); the eval
  scripts share `scripts/sweep_eval.py` (`run_eval`, 2-GPU pool, resumable).
- **Diagnosis notebooks (§2):** `notebooks/{eq20_failure_analysis,eval_loss_floor_irreducible,bf16_vs_fp32_reasoning}.ipynb`.
  checkpoint — included for the technique, not as a finding about our model.
