# Status — Phase 2 (2026-06-25)

*Snapshot of Phase-2 progress on branch `phase_2`. Pairs with
[`phase-2-sprint-plan.md`](./phase-2-sprint-plan.md) and
[`phase-2-hypernet-sizing-options.md`](./phase-2-hypernet-sizing-options.md).*

## TL;DR

**Sprints 1 & 2 done & committed; the split is re-locked & committed; Sprint 3
(reconstruction warmup) is mid-flight and blocked on one decision: switch the
output parameterization `vera → lowrank`.** The reason: a diagnostic showed VeRA
*cannot reconstruct* a target ΔW (frozen random basis), while low-rank can. All
code is committed and pushed; the working tree is clean; nothing is running; GPU
is free.

---

## Sprint progress

| Sprint | State | Notes |
|---|---|---|
| S1 — tiny-model plumbing + W&B contract | ✅ done | `RunLogger` wired into the loop; online W&B working (after the entity fix) |
| S2 — architecture + output parameterization | ✅ done | size report → table T3 (size half) |
| **split re-lock** (S3 prerequisite) | ✅ done | curated 400/10/30, `lock_hash=ca213edbccb6e5b8` |
| S3 — reconstruction warmup | ⏳ blocked | runs end-to-end but VeRA can't reconstruct → needs `lowrank` + re-run |
| S4 — SFT meta-train (the gate run) | ⬜ pending | needs S3 warm-start + a VRAM pre-flight |
| S5 — baselines + held-out eval + the gate | ⬜ pending | new `phase2_eval_gate.py` (T1–T6, F1–F6) |
| S6 — findings + artifact | ⬜ pending | `docs/phase-2-findings.md` + W&B report |

---

## Commands run this session

| # | Command (abbreviated) | Status | Output / result |
|---|---|---|---|
| 1 | `pytest tests/` after wiring `RunLogger` into the loop | ✅ | 91 green |
| 2 | `phase2_meta_train.py --config tiny-plumbing --wandb online` (S1 verify) | ⚠️→fixed | W&B `permission denied`; fell back offline |
| 3 | minimal `wandb.init(entity="doku88")` probe | ✅ | **Root cause:** default entity is team `ctorl`; `doku88` works → run synced |
| 4 | `phase2_size_report.py` (S2, table T3) | ✅ (after 2 fixes) | VeRA 55.7M / low-rank 151M / full 2.65B; W&B run `e692ef0b` |
| 5 | `phase2_relock_split.py` (dry-run) | ✅ (after 2 curation fixes) | 19 format pairs / 249 translation / 15 domain candidates |
| 6 | `phase2_relock_split.py --lock` | ✅ | wrote `heldout_split.yaml`, `lock_hash=ca213edbccb6e5b8`, 400/10/30, leakage guard OK |
| 7 | S3 recon smoke (12 steps, real path) | ❌→fixed | **bug 1**: generator index tensors on CPU vs GPU embeddings |
| 8 | S3 smoke re-run | ❌→fixed | **bug 2**: recon target on CPU vs generated on GPU |
| 9 | S3 smoke re-run | ⚠️→fixed | **bug 3**: loss = `0.00000` (vanishing — mean-L1 over ~4M tiny elements) |
| 10 | S3 smoke (25 steps, relative-Frobenius loss) | ✅ runs | loss meaningful (1.0 init) but noisy per-step (each step a different task) |
| 11 | Full S3 warmup launched (3000 steps, bg) | ⛔ killed | **stuck at loss 1.0000** at ~1450 steps — not learning |
| 12 | GPU diagnostic: single-target overfit + grad check | ✅ key finding | VeRA: `b_scale` moves but `d_scale`/trunk grad ≈ 0; loss 1.0→0.9999 |
| 13 | Parameterization comparison (overfit one target) | ✅ decisive | **VeRA 1.0→0.9999 (can't fit); low-rank 1.0→0.197 (fits); full OOM** |

**W&B runs created** (all at `wandb.ai/doku88/lora-lab-phase2`): S1 plumbing; S2
size-report (`e692ef0b`); S3 smokes + the killed warmup (`g9a7jkci`).

---

## The key finding — VeRA cannot reconstruct; switch to low-rank

Single real library-adapter target, overfit on GPU:

| Parameterization | Params | Single-target overfit (relative ΔW error) | Verdict |
|---|--:|---|---|
| **VeRA** (current committed default) | 55.8M | 1.0000 → **0.9999** | ❌ cannot fit — frozen random A/B basis |
| **low-rank** | 151.6M | 1.0000 → **0.1966** | ✅ fits (generates real A/B via a bottleneck) |
| full | 2.65B | — | ❌ OOM (> 32 GB with Adam states) |

**Why:** VeRA's A and B are *frozen random* matrices; the hypernetwork emits only
scaling vectors `(d, b)` that reweight those fixed directions — it can't reshape
them to match a specific adapter. The diagnostic confirmed it: `b_scale` receives
gradient and moves, but `d_scale` and the conditioning trunk get ≈0 gradient, and
even fully overfitting *one* target moves the error by 0.0001. This is exactly the
limitation the sizing doc named ("can only reweight fixed random directions, can't
reshape them"), and its recommended *cheapest upgrade* is `lowrank ≈ T2L-M`.

**This is a "plumbing before scale" win:** the committed `parameterization: vera`
default would have trained a hypernetwork that can't represent adapters — failing
the warmup *and* cascading into a dead SFT gate — caught in a 5-minute diagnostic
instead of a 7-hour run.

**Pending decision:** switch `configs/phase2/{recon-warmup,sft-mistral}.yaml`
from `vera` to `lowrank`, then re-run S3 (should converge) and verify low-rank's
151M trainable params fit the SFT backprop-through-base on 32 GB (S4 pre-flight).

---

## Code changes — committed this session

| Commit | Sprint | What |
|---|---|---|
| `015e985` | S1 | wire `RunLogger` into `meta_train` (per-step loss/lr/grad-norm/VRAM → `metrics.jsonl` + W&B); `logging.py` adapter; entrypoint `--wandb`/`--stage`; +2 tests |
| `e08a369` | S1 fix | `wandb_entity="doku88"` in `HyperConfig` (fixes the 403) |
| `ac51b02` | S2 | `phase2_size_report.py` → table T3 (size half); +1 test; W&B logging |
| `c6fc830` | split | `phase2_relock_split.py` + re-locked `heldout_split.yaml` (curated 400/10/30) |
| `92cb64f` | S3 fixes | 3 debugging fixes (below) + gitignore `results/phase2/*.log` |

### The 3 S3 debugging fixes (commit `92cb64f`)

All surfaced by running the *real* reconstruction path on GPU (CPU tests masked
all three):

| File | Fix |
|---|---|
| `src/lora_lab/hypernet/model.py` | `HyperLoRAGenerator.forward` builds layer/module index tensors on `task_emb.device` (was CPU vs cuda embeddings) |
| `src/lora_lab/hypernet/meta_train.py` | move reconstruction target tensors to the run device before the loss |
| `src/lora_lab/hypernet/recon.py` | reconstruction loss: mean-L1 → **relative Frobenius error** `‖ΔW_g−ΔW_t‖/‖ΔW_t‖` (non-vanishing gradient) |

---

## Planned commands — numbered, mapped to sprints

> **#1 is gated on the `vera → lowrank` decision.** Everything below assumes yes.

| # | Command | Sprint | Purpose |
|---|---|---|---|
| 1 | Edit `configs/phase2/{recon-warmup,sft-mistral}.yaml`: `vera → lowrank`; `pytest tests/`; **commit** the switch | S3 (unblock) | adopt the only parameterization that can reconstruct |
| 2 | `python scripts/phase2_meta_train.py --config configs/phase2/recon-warmup.yaml --allow-gpu --wandb online --stage S3-recon-warmup` | **S3** | reconstruction warmup (~20–30 min); confirm loss converges; save `hypernet.pt` artifact → **W&B verify** + S3 commit |
| 3 | single-batch SFT memory pre-flight: 4-bit Mistral + low-rank adapter, 1 step, assert peak VRAM ≤ 32 GB | S4 (pre-flight) | confirm 151M trainable + base-backprop fits before the long run |
| 4 | `python scripts/phase2_meta_train.py --config configs/phase2/sft-mistral.yaml --allow-gpu --wandb online --stage S4-sft` (warm-starts from #2) | **S4** | the gate run — SFT meta-train (~5–7.5 hr); F4 VRAM trace + checkpoint → **W&B verify** + S4 commit |
| 5 | new `scripts/phase2_eval_gate.py`: eval generated / oracle / base / retrieval on the 30 held-out tasks → tables T1–T2, T5–T6, figs F1–F2, F5–F6; gate verdict | **S5** | the critical gate (generated > retrieval); + S5 commit |
| 6 | write `docs/phase-2-findings.md` + W&B report | **S6** | findings, gate verdict, reproducibility → **W&B verify** + S6 commit |

---

## Locked split — for reference

`configs/phase1/heldout_split.yaml`, `lock_hash=ca213edbccb6e5b8`: **train 400 ·
val 10 · held-out 30 · reserved 597** (of the 1,037 gate-passing Phase-1 library
tasks). Held-out axes: **format transfer** (15 — hold out the generation form,
train the classification form), **language transfer** (8 — hold out a direction,
train its reverse), **domain transfer** (7 — hold out twitter/financial/bengali/
poem/pec sentiment, train amazon/yelp/sent140). Every held-out task's trained
partner is forced into train; leakage guard passes (held-out ∩ train = ∅).
