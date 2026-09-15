# Phase-4.5 execution audit

## Protocol and completeness

- Fixed cohort: `datasets/phase3/scenario_bank_512.pt`; all runs used the authoritative bank hash recorded in the manifests.
- Models: seed1 C and seed2 C, 512 scenarios per run, `num_envs=64`, `cuda:0`, controller observation noise OFF, and explicit environment randomization OFF.
- Sweep: beta = 0.5, 0.75, 1.0; 3 independent processes per beta per training seed (18 sweep runs).
- Completeness check: 18/18 sweep runs valid; no validation errors.

## Observational nominal baseline

| training seed | safe success | collision | `V_filter_gt` | `V_bound_gt` | `V_track_gt` |
|---:|---:|---:|---:|---:|---:|
| 1 | 34.38% | 56.25% | 0.51% | 0.00% | 1.69% |
| 2 | 39.84% | 55.27% | 0.46% | 0.00% | 1.63% |

The observational runs above are the earlier beta=0.5 nominal executions; the sweep below provides the repeated counterfactual filter comparison.

## Main result by filter beta

| beta | safe success | collision | `V_filter_gt` | `V_bound_gt` | `V_track_gt` | filter delta | tracking error post |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 36.36% | 55.40% | 0.50% | 0.00% | 1.70% | 0.0339 | 0.1434 |
| 0.75 | 35.77% | 56.15% | 0.19% | 0.00% | 1.70% | 0.0128 | 0.1479 |
| 1 | 36.85% | 55.31% | 0.00% | 0.00% | 1.74% | 0.0000 | 0.1503 |

### Matched terminal windows

The window comparison is descriptive and outcome-conditioned; it is not itself a counterfactual intervention.

| beta | collision-window `V_filter_gt` | success-window `V_filter_gt` | collision-window `V_track_gt` | success-window `V_track_gt` | collision-window Δr_filter | success-window Δr_filter |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.60% | 0.10% | 2.06% | 0.45% | -0.0061 | -0.0009 |
| 0.75 | 0.25% | 0.04% | 2.17% | 0.48% | -0.0020 | -0.0003 |
| 1 | 0.00% | 0.00% | 2.12% | 0.48% | 0.0000 | 0.0000 |

## Bottleneck grading

| component | residual magnitude | causal confidence from this phase | interpretation |
|---|---|---|---|
| POST-CBF filter | HIGH at beta=0.5; eliminated at beta=1.0 | HIGH | The only stage with a clean evaluator-only counterfactual sweep and a corresponding GT residual reduction. |
| COMMAND BOUND | LOW | LOW | Bound clipping is observable but `V_bound_gt` is orders of magnitude below tracking violation. |
| LOCOMOTION TRACKING | HIGH residual incidence; about 1.7% `V_track_gt` | MED | It persists after removing the filter effect; outcome causality still needs a separate tracking intervention. |
| CBF formulation / estimator | LOW evidence as dominant execution cause | UNRESOLVED for latent estimator error | Raw clamp and shadow repair are zero, but this audit does not prove estimator generalization. |


## Causal interpretation

1. **POST-CBF filter:** high-confidence causal candidate. With beta=0.5, the filter creates measurable GT residual violations; with beta=1.0, the filter stage is exactly the raw-clipped CBF output and `V_filter_gt` is zero by construction. The sweep therefore isolates a real execution-stage effect rather than only a diagnostic correlation.
2. **Filter sensitivity:** the mean `V_filter_gt` reduction from beta=0.5 to beta=1.0 is 100.00% when both aggregate rows are available; beta=0.75 is an intermediate condition.
3. **COMMAND BOUND:** low-frequency contributor. Raw CBF clipping is zero in every sweep run; nav-bound clipping occurs, but `V_bound_gt` remains at the order of 10^-6 to 10^-5 and is much smaller than tracking violations.
4. **LOCOMOTION TRACKING:** persistent independent bottleneck. `V_track_gt` stays around the 1.6–1.8% range across beta values, including beta=1.0, so removing the low-pass filter does not remove the low-level tracking residual.
5. **CBF formulation / estimator:** no evidence here for raw CBF saturation or a learned-vs-GT formulation mismatch as the dominant execution-stage cause. Raw-clamp events and shadow repairs are zero; this is an execution audit result, not a proof that the estimator is perfect.

## Requested Phase-4.5 decisions

- **A — Nominal protocol:** use controller observation noise OFF and retain fixed-cohort repeated trials. Record the production filter beta explicitly in the manifest.
- **B — Robustness protocol:** keep controller observation noise ON as a separate protocol and report it independently; do not mix it with nominal causal comparisons.
- **C — Filter decision:** treat beta=1.0 as the clean no-filter diagnostic baseline. Do not change the training default or deployment filter solely from this audit because aggregate success/collision rates are not monotonic across beta.
- **D — Tracking priority:** make locomotion command tracking the next engineering target; the residual remains at the same order for all beta values.
- **E — CBF/estimator:** do not retrain or reformulate the CBF from this evidence alone; preserve the current learned-vs-GT and shadow-reprojection diagnostics for a later estimator-focused test.
- **F — Reporting:** report mean and repeat spread across independent fixed-cohort trials, with GPU-PhysX nondeterminism treated as expected residual variance.

## Artifacts

- Run-level table: `training/legged_gym/logs/go2_pos_dynamic/phase4_5_execution_audit/analysis/filter_sweep_summary.csv`
- Episode-level table: `training/legged_gym/logs/go2_pos_dynamic/phase4_5_execution_audit/analysis/execution_episode_summary.csv`
- Machine-readable aggregate: `training/legged_gym/logs/go2_pos_dynamic/phase4_5_execution_audit/analysis/execution_stage_summary.json`
