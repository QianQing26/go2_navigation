# Phase-4 Residual Failure Analysis + Oracle Upper-Bound Study

Validated training seeds for Phase-4: **1, 2**. Seed3 is excluded because its earlier evaluation was confirmed invalid.

The study uses one fixed 512-scenario bank, frozen policies, six drift modes, and three repeats per mode and seed (36 oracle records). Learned and oracle modes share actor observations, policy weights, rays, terrain, and scenario order; oracle GT drift is refreshed at 10 Hz and zero-order-held for five 50 Hz control steps. Replay is disabled for evaluation and no oracle policy is trained.

## Oracle fixed-policy results

| Seed | Mode | Safe-success mean | Collision mean | Repeats |
|---:|---|---:|---:|---:|
| 1 | predictive_learned | 0.3464 | 0.5684 | 3 |
| 1 | oracle_gt_0p1 | 0.3477 | 0.5625 | 3 |
| 1 | oracle_gt_0p2 | 0.3496 | 0.5579 | 3 |
| 1 | oracle_gt_0p3 | 0.3483 | 0.5638 | 3 |
| 1 | oracle_gt_0p5 | 0.3307 | 0.5775 | 3 |
| 1 | oracle_gt_multi | 0.3607 | 0.5573 | 3 |
| 2 | predictive_learned | 0.3880 | 0.5573 | 3 |
| 2 | oracle_gt_0p1 | 0.3887 | 0.5573 | 3 |
| 2 | oracle_gt_0p2 | 0.3887 | 0.5566 | 3 |
| 2 | oracle_gt_0p3 | 0.3939 | 0.5527 | 3 |
| 2 | oracle_gt_0p5 | 0.3822 | 0.5671 | 3 |
| 2 | oracle_gt_multi | 0.3958 | 0.5540 | 3 |

## Headroom

- Estimator headroom (GT-0.1 vs learned): SR `0.0010`, CR reduction `0.0029`.
- Horizon headroom (GT-multi vs GT-0.1): SR `0.0101`, CR reduction `0.0042`.
- Oracle matrix completeness: `36/36 records`.
- Failure taxonomy counts: `{"F1_estimator_decision_miss": 146, "F2_estimator_magnitude_underestimate": 277, "F3_short_horizon": 5, "F4_execution_control_authority": 577, "F5_geometry_or_residual": 0}`.

## Continuous pre-collision metrics

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| false_safe_fraction | 0.0844 | 0.1729 | 0.0000 | 1.0000 |
| mean_abs_drift_error | 0.1507 | 0.2340 | 0.0000 | 1.9901 |
| max_abs_drift_error | 0.5330 | 0.7372 | 0.0001 | 10.4714 |
| dangerous_drift_mae | 0.2221 | 0.5135 | 0.0000 | 9.7239 |
| optimistic_danger_mean | 0.2001 | 0.5181 | 0.0000 | 9.7239 |
| optimistic_danger_max | 0.4154 | 0.6366 | 0.0000 | 10.4714 |
| learned_detection_delay_s | 0.0505 | 0.2373 | -1.3000 | 1.8000 |
| mean_oracle_extra_intervention_0p1 | 0.0344 | 0.0873 | -0.0556 | 0.9290 |
| max_oracle_extra_intervention_0p1 | 0.1653 | 0.2809 | -0.0058 | 4.9772 |
| clip_fraction | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| max_clip_delta | 0.3935 | 0.1696 | 0.0151 | 0.8958 |
| mean_tracking_error | 0.1907 | 0.0996 | 0.0512 | 0.8778 |
| max_tracking_error | 0.5978 | 0.3308 | 0.1309 | 1.7581 |
| mean_robot_speed_pre_collision | 0.5947 | 0.2417 | 0.0560 | 1.1809 |
| min_obstacle_distance | 0.7713 | 0.3525 | 0.5243 | 5.4744 |
| warning_time_0p1 | 1.3653 | 0.5703 | 0.0400 | 2.0000 |
| warning_time_0p2 | 1.3702 | 0.5609 | 0.0200 | 2.0000 |
| warning_time_0p3 | 1.3825 | 0.5475 | 0.1200 | 2.0000 |
| warning_time_0p5 | 1.3970 | 0.5333 | 0.1400 | 2.0000 |

## Failure Attribution vs Oracle Causality

The taxonomy is retained as a non-exclusive descriptive summary. F4 is an execution/control signal, not proof of causality; the paired oracle results, not heuristic labels alone, determine the causal interpretation.

## Bottleneck assessment

- ESTIMATOR BOTTLENECK: **LOW**
- HORIZON BOTTLENECK: **LOW**
- CONTROL/EXECUTION BOTTLENECK: **HIGH**
- GEOMETRY/CBF BOTTLENECK: **LOW**

## Recommended next direction

**C. Controller / locomotion-aware safety model**

This recommendation is descriptive and based on the fixed-policy oracle gap; no oracle-trained policy was evaluated.
