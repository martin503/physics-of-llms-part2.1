
| size | feature |
|:----:|---------|
| medium | (highest priority) successfully train v-probe on dep(A, B) task |
| small | integrate probe training with WandB to track progress, see visual learning behavior and keep a record of past runs. |
| small | integrate probe training/ testing with inspect to easily analyze results. |
| medium | Add custom visualisation to compare probe predictions with ground truth values. (First for dep(A, B)) |
| large | Add statistical analysis to gain more trustworthy results. (First for dep(A, B)) |
| large | add remaining probes from paper, documented in [[paper_summary.md]]</br> Look at the paper to understand correct input format and probe location. |
| medium | re-run the nece V-probe (+ random control) on the fixed HF model — all pre-fix probe results are invalid (garbage `inv_freq` before the persistent-buffer fix). |
| small | linear-probe dep baseline: cache the end-of-problem-description read position in `extract.py` (currently nece-only). |
| small | delete or re-save the old local checkpoint `final_models/gpt2-rope-igsm/final` (loads with garbage `inv_freq`; `load_lm` now rejects it loudly). |
| small | reconcile branches: `eval` still has the `inv_freq` bug; the fix lives on `qknorm` (4957741) and `probes`. Consider merging so no branch silently loads corrupted models. |
