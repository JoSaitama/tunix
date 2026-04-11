# Develop History Timeline

This folder preserves raw per-node `develop.md` snapshots under `raw/` and provides a merged chronological index here.

## Sources
- `raw/node-0.md`
- `raw/node-1.md`
- `raw/node-2.md`
- `raw/node-3.md`
- `raw/node-4.md`
- `raw/node-v5.md`

## Notes on interpretation
- `raw/` keeps the original node-local histories for auditability.
- This timeline merges duplicated entries by date/topic and records which node snapshots contain them.
- Several DPO-era snapshots are effectively shared across `node-2`, `node-3`, `node-4`, and `node-v5`; the duplicate raw files are still preserved unchanged.

---

## 2026-03-09 — DeepScaler wrapper and prompt-safety iteration
**Source nodes:** `node-0`

Key work captured in `raw/node-0.md`:
- DeepScaler skip-overlong-prompt handling
- Prompt-filter scope clarification
- Overlong-prompt smoke testing
- Wrapper default adjustments around total generation steps

Why it matters:
- This is the main early DeepScaler wrapper / prompt-safety development line preserved in the raw history.

---

## 2026-03-10 — DeepScaler DBC wrapper and command/default cleanup
**Source nodes:** `node-0`

Key work captured in `raw/node-0.md`:
- G=2 / G=4 / G=8 training-time scaling estimates
- DBC command clarification
- G=8 command note and wrapper default updates
- Dedicated DeepScaler DBC wrapper added
- Pushed-content summary for the wrapper-related changes

Why it matters:
- This is the clearest historical record of the older DeepScaler wrapper engineering line.

---

## 2026-03-16 — Initial Qwen2.5 UltraFeedback SFT → DPO workflow line
**Source nodes:** `node-2`, `node-3`, `node-4`, `node-v5`

Key work captured in the raw snapshots:
- Interpreting overfitting in aligned-model DPO artifacts
- Qwen2.5-1.5B UltraFeedback SFT → DPO + DBC implementation
- README availability checks for the SFT → DPO flow

Why it matters:
- This marks the start of the DPO workflow line that later fed the v4 multi-node experiments.

---

## 2026-03-17 — SFT run, exported model, DPO smoke validation, and local loading fixes
**Source nodes:** `node-2`, `node-3`, `node-4`, `node-v5`

Key work captured in the raw snapshots:
- Full Qwen2.5-1.5B UltraFeedback SFT run
- Exported-model generation sanity checks
- SFT checkpoint cleanup
- Reproducibility note updates
- Experiment-note renaming to `paper_experiment.md`
- DPO-from-SFT smoke validation and local INTERNAL model-loading fix
- DPO smoke re-validation under sandbox restrictions
- README command pointers
- DPO hyperparameter summary review
- Runtime environment clarification

Why it matters:
- This is the shared DPO bootstrap history that appears across the v4-node line and the copied raw snapshot on `node-v5`.

---

## 2026-04-08 — DeepScaler resume, disk, and runtime stabilization
**Source nodes:** `node-1`

Key work captured in `raw/node-1.md`:
- Disk-usage inspection and reclamation during resumed training work
- Assessment of whether resumed runs would hit disk limits again
- Clarification around checkpoint retention / final-checkpoint-only saving
- Restoring default DeepScaler model and dataset snapshots
- Fixing resumed fast-path crashes in async metrics buffering
- Clarifying TensorBoard continuity under resume

Why it matters:
- This is the main recent DeepScaler runtime / resume stabilization line.

---

## 2026-04-09 — Upstream DeepScaler survey
**Source nodes:** `node-1`

Key work captured in `raw/node-1.md`:
- Survey of upstream `google/tunix` DeepScaler-related changes
- Classification of upstream items by importance and likely merge value
- Comparison notes for selectively absorbing correctness fixes without blindly switching to upstream

Why it matters:
- This is the best raw record of the recent DeepScaler upstream-comparison work.

---

## Snapshot layout summary
- `node-0`: older DeepScaler wrapper / DBC / prompt-safety history
- `node-1`: recent DeepScaler runtime, resume, disk, and upstream survey history
- `node-2`, `node-3`, `node-4`: shared DPO SFT → DPO experiment history
- `node-v5`: preserved raw snapshot for comparison, currently overlapping heavily with the DPO-era shared history
