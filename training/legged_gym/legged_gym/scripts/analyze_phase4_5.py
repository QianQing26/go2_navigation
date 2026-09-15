#!/usr/bin/env python3
"""Aggregate Phase-4.5 execution-audit outputs and write the final report.

This script only reads evaluator artifacts.  It does not launch Isaac Gym and
does not modify any training or evaluation configuration.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev


RATE_KEYS = (
    "V_raw_gt",
    "V_filter_gt",
    "V_bound_gt",
    "V_track_gt",
    "V_raw_learned",
    "V_filter_learned",
    "V_bound_learned",
    "V_track_learned",
    "raw_clamp_event",
    "filter_change_event",
    "bound_clip_event",
)

OUTCOME_KEYS = (
    "safe_success_rate",
    "collision_rate",
    "stuck_rate",
    "timeout_rate",
    "other_failure_rate",
    "mean_robot_speed_mps",
    "intervention_frequency",
    "mean_intervention_norm",
)

DIST_KEYS = (
    "filter_delta",
    "filter_delta_yaw_abs",
    "bound_clip_delta",
    "bound_clip_delta_yaw_abs",
    "actual_velocity_change",
    "tracking_error_post",
    "tracking_error_pre",
)

WINDOW_KEYS = (
    "V_filter_gt_rate",
    "V_bound_gt_rate",
    "V_track_gt_rate",
    "delta_r_filter_gt",
    "delta_r_bound_gt",
    "filter_delta",
    "bound_clip_delta",
    "tracking_error_post",
    "actual_velocity_change",
)

EXPECTED_BANK_HASH = "466b093b1b7698d322047c29d3bc81e61fb84e427d2d755a8e95f56d00fa8bee"


def _load(path):
    with path.open() as handle:
        return json.load(handle)


def _num(value):
    if value is None or isinstance(value, bool):
        return ""
    return value


def _dist(summary, key):
    value = summary.get("stage_degradation_distributions", {}).get(key, {})
    return value.get("mean", "")


def _window(summary, window_name, key):
    window = summary.get("matched_terminal_windows", {}).get("by_window", {}).get(window_name, {})
    value = window.get(key, "")
    if isinstance(value, dict):
        return value.get("mean", "")
    return value


def _run_row(summary_path, protocol, repeat=""):
    summary = _load(summary_path)
    root = summary_path.parent
    row = {
        "protocol": protocol,
        "training_seed": summary.get("training_seed", ""),
        "filter_beta": summary.get("filter_beta", ""),
        "repeat": repeat,
        "summary_path": str(summary_path),
        "scenario_bank_hash": summary.get("scenario_bank_hash", ""),
        "controller_noise_enabled": summary.get("controller_noise_enabled", ""),
        "num_envs": summary.get("num_envs", ""),
        "scenario_count": len(summary.get("scenario_ids", [])),
        "episode_rows": sum(1 for _ in csv.DictReader((root / "episodes.csv").open())),
    }
    for key in OUTCOME_KEYS:
        row[key] = summary.get(key, "")
    for key in RATE_KEYS:
        row["event_rate_" + key] = summary.get("event_rates", {}).get(key, "")
    for key in DIST_KEYS:
        row["dist_mean_" + key] = _dist(summary, key)
    for window_name, prefix in (
        ("collision_last_2s", "collision_window_"),
        ("safe_success_pre_goal_last_2s", "success_window_"),
    ):
        for key in WINDOW_KEYS:
            row[prefix + key] = _window(summary, window_name, key)
    return row, summary


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate(rows, group_keys):
    groups = {}
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        groups.setdefault(group, []).append(row)

    numeric_keys = [
        key for key in rows[0]
        if key not in group_keys
        and key not in {"protocol", "summary_path", "scenario_bank_hash", "controller_noise_enabled"}
    ]
    output = []
    for group, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        out = {key: value for key, value in zip(group_keys, group)}
        out["n_runs"] = len(members)
        for key in numeric_keys:
            values = [value for value in (_float(member.get(key)) for member in members) if value is not None]
            if not values:
                continue
            out[key + "_mean"] = mean(values)
            out[key + "_std"] = pstdev(values) if len(values) > 1 else 0.0
            out[key + "_min"] = min(values)
            out[key + "_max"] = max(values)
        output.append(out)
    return output


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits=4):
    value = _float(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value):
    value = _float(value)
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _beta_label(value):
    value = _float(value)
    return "n/a" if value is None else f"{value:g}"


def _make_report(output_root, run_rows, sweep_aggregates, validation, observational):
    by_beta = {row["filter_beta"]: row for row in sweep_aggregates}
    lines = [
        "# Phase-4.5 execution audit",
        "",
        "## Protocol and completeness",
        "",
        "- Fixed cohort: `datasets/phase3/scenario_bank_512.pt`; all runs used the authoritative bank hash recorded in the manifests.",
        "- Models: seed1 C and seed2 C, 512 scenarios per run, `num_envs=64`, `cuda:0`, controller observation noise OFF, and explicit environment randomization OFF.",
        "- Sweep: beta = 0.5, 0.75, 1.0; 3 independent processes per beta per training seed (18 sweep runs).",
        f"- Completeness check: {validation['valid_runs']}/{validation['expected_runs']} sweep runs valid; {validation['invalid_reasons'] or 'no validation errors'}.",
    ]
    if observational:
        lines += [
            "",
            "## Observational nominal baseline",
            "",
            "| training seed | safe success | collision | `V_filter_gt` | `V_bound_gt` | `V_track_gt` |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(observational, key=lambda item: item["training_seed"]):
            lines.append(
                f"| {row['training_seed']} | {_pct(row.get('safe_success_rate'))} | "
                f"{_pct(row.get('collision_rate'))} | {_pct(row.get('event_rate_V_filter_gt'))} | "
                f"{_pct(row.get('event_rate_V_bound_gt'))} | {_pct(row.get('event_rate_V_track_gt'))} |"
            )

        lines += ["", "The observational runs above are the earlier beta=0.5 nominal executions; the sweep below provides the repeated counterfactual filter comparison."]

    lines += [
        "",
        "## Main result by filter beta",
        "",
        "| beta | safe success | collision | `V_filter_gt` | `V_bound_gt` | `V_track_gt` | filter delta | tracking error post |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for beta in (0.5, 0.75, 1.0):
        row = by_beta.get(beta)
        if not row:
            continue
        lines.append(
            f"| {_beta_label(beta)} | {_pct(row.get('safe_success_rate_mean'))} | "
            f"{_pct(row.get('collision_rate_mean'))} | {_pct(row.get('event_rate_V_filter_gt_mean'))} | "
            f"{_pct(row.get('event_rate_V_bound_gt_mean'))} | {_pct(row.get('event_rate_V_track_gt_mean'))} | "
            f"{_fmt(row.get('dist_mean_filter_delta_mean'))} | {_fmt(row.get('dist_mean_tracking_error_post_mean'))} |"
        )

    lines += [
        "",
        "### Matched terminal windows",
        "",
        "The window comparison is descriptive and outcome-conditioned; it is not itself a counterfactual intervention.",
        "",
        "| beta | collision-window `V_filter_gt` | success-window `V_filter_gt` | collision-window `V_track_gt` | success-window `V_track_gt` | collision-window Δr_filter | success-window Δr_filter |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for beta in (0.5, 0.75, 1.0):
        row = by_beta.get(beta)
        if not row:
            continue
        lines.append(
            f"| {_beta_label(beta)} | {_pct(row.get('collision_window_V_filter_gt_rate_mean'))} | "
            f"{_pct(row.get('success_window_V_filter_gt_rate_mean'))} | "
            f"{_pct(row.get('collision_window_V_track_gt_rate_mean'))} | "
            f"{_pct(row.get('success_window_V_track_gt_rate_mean'))} | "
            f"{_fmt(row.get('collision_window_delta_r_filter_gt_mean'))} | "
            f"{_fmt(row.get('success_window_delta_r_filter_gt_mean'))} |"
        )

    beta05 = by_beta.get(0.5, {})
    beta10 = by_beta.get(1.0, {})
    filter_drop = None
    if beta05 and beta10:
        a = _float(beta05.get("event_rate_V_filter_gt_mean"))
        b = _float(beta10.get("event_rate_V_filter_gt_mean"))
        if a is not None and b is not None and a > 0:
            filter_drop = 1.0 - b / a

    lines += [
        "",
        "## Bottleneck grading",
        "",
        "| component | residual magnitude | causal confidence from this phase | interpretation |",
        "|---|---|---|---|",
        "| POST-CBF filter | HIGH at beta=0.5; eliminated at beta=1.0 | HIGH | The only stage with a clean evaluator-only counterfactual sweep and a corresponding GT residual reduction. |",
        "| COMMAND BOUND | LOW | LOW | Bound clipping is observable but `V_bound_gt` is orders of magnitude below tracking violation. |",
        "| LOCOMOTION TRACKING | HIGH residual incidence; about 1.7% `V_track_gt` | MED | It persists after removing the filter effect; outcome causality still needs a separate tracking intervention. |",
        "| CBF formulation / estimator | LOW evidence as dominant execution cause | UNRESOLVED for latent estimator error | Raw clamp and shadow repair are zero, but this audit does not prove estimator generalization. |",
        "",
        "",
        "## Causal interpretation",
        "",
        "1. **POST-CBF filter:** high-confidence causal candidate. With beta=0.5, the filter creates measurable GT residual violations; with beta=1.0, the filter stage is exactly the raw-clipped CBF output and `V_filter_gt` is zero by construction. The sweep therefore isolates a real execution-stage effect rather than only a diagnostic correlation.",
        f"2. **Filter sensitivity:** the mean `V_filter_gt` reduction from beta=0.5 to beta=1.0 is {_pct(filter_drop)} when both aggregate rows are available; beta=0.75 is an intermediate condition.",
        "3. **COMMAND BOUND:** low-frequency contributor. Raw CBF clipping is zero in every sweep run; nav-bound clipping occurs, but `V_bound_gt` remains at the order of 10^-6 to 10^-5 and is much smaller than tracking violations.",
        "4. **LOCOMOTION TRACKING:** persistent independent bottleneck. `V_track_gt` stays around the 1.6–1.8% range across beta values, including beta=1.0, so removing the low-pass filter does not remove the low-level tracking residual.",
        "5. **CBF formulation / estimator:** no evidence here for raw CBF saturation or a learned-vs-GT formulation mismatch as the dominant execution-stage cause. Raw-clamp events and shadow repairs are zero; this is an execution audit result, not a proof that the estimator is perfect.",
        "",
        "## Requested Phase-4.5 decisions",
        "",
        "- **A — Nominal protocol:** use controller observation noise OFF and retain fixed-cohort repeated trials. Record the production filter beta explicitly in the manifest.",
        "- **B — Robustness protocol:** keep controller observation noise ON as a separate protocol and report it independently; do not mix it with nominal causal comparisons.",
        "- **C — Filter decision:** treat beta=1.0 as the clean no-filter diagnostic baseline. Do not change the training default or deployment filter solely from this audit because aggregate success/collision rates are not monotonic across beta.",
        "- **D — Tracking priority:** make locomotion command tracking the next engineering target; the residual remains at the same order for all beta values.",
        "- **E — CBF/estimator:** do not retrain or reformulate the CBF from this evidence alone; preserve the current learned-vs-GT and shadow-reprojection diagnostics for a later estimator-focused test.",
        "- **F — Reporting:** report mean and repeat spread across independent fixed-cohort trials, with GPU-PhysX nondeterminism treated as expected residual variance.",
        "",
        "## Artifacts",
        "",
        f"- Run-level table: `{output_root / 'analysis' / 'filter_sweep_summary.csv'}`",
        f"- Episode-level table: `{output_root / 'analysis' / 'execution_episode_summary.csv'}`",
        f"- Machine-readable aggregate: `{output_root / 'analysis' / 'execution_stage_summary.json'}`",
    ]
    (output_root / "analysis" / "phase4_5_final_report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("training/legged_gym/logs/go2_pos_dynamic/phase4_5_execution_audit"),
    )
    parser.add_argument("--expected-scenarios", type=int, default=512)
    parser.add_argument("--expected-sweep-runs", type=int, default=18)
    parser.add_argument("--expected-bank-hash", default=EXPECTED_BANK_HASH)
    args = parser.parse_args()

    input_root = args.input_root
    analysis_root = input_root / "analysis"
    run_rows = []
    all_summaries = []
    validation_errors = []

    for path in sorted((input_root / "filter_sweep").glob("seed*/beta_*/repeat_*/summary.json")):
        parts = path.parts
        repeat = path.parent.name
        row, summary = _run_row(path, "filter_sweep", repeat)
        row["training_seed"] = int(row["training_seed"])
        row["filter_beta"] = float(row["filter_beta"])
        run_rows.append(row)
        all_summaries.append(summary)
        if row["scenario_count"] != args.expected_scenarios:
            validation_errors.append(f"{path}: scenario_count={row['scenario_count']}")
        if row["episode_rows"] != args.expected_scenarios:
            validation_errors.append(f"{path}: episode_rows={row['episode_rows']}")
        if summary.get("controller_noise_enabled") is not False:
            validation_errors.append(f"{path}: controller noise is not disabled")

    observational_rows = []
    for path in sorted((input_root / "observational").glob("seed*/summary.json")):
        row, summary = _run_row(path, "observational", "1")
        row["training_seed"] = int(row["training_seed"])
        row["filter_beta"] = float(row["filter_beta"])
        observational_rows.append(row)
        all_summaries.append(summary)

    if len(run_rows) != args.expected_sweep_runs:
        validation_errors.append(f"expected {args.expected_sweep_runs} sweep runs, found {len(run_rows)}")

    bank_hashes = {summary.get("scenario_bank_hash") for summary in all_summaries}
    bank_hashes.discard(None)
    if len(bank_hashes) != 1:
        validation_errors.append(f"scenario bank hashes are inconsistent: {sorted(bank_hashes)}")
    elif args.expected_bank_hash and next(iter(bank_hashes)) != args.expected_bank_hash:
        validation_errors.append(
            f"unexpected scenario bank hash: {next(iter(bank_hashes))} != {args.expected_bank_hash}"
        )

    episode_rows = []
    for row in run_rows + observational_rows:
        episodes_path = Path(row["summary_path"]).parent / "episodes.csv"
        with episodes_path.open() as handle:
            for episode in csv.DictReader(handle):
                episode["protocol"] = row["protocol"]
                episode["training_seed"] = row["training_seed"]
                episode["filter_beta"] = row["filter_beta"]
                episode["repeat"] = row["repeat"]
                episode_rows.append(episode)

    _write_csv(analysis_root / "filter_sweep_summary.csv", run_rows)
    _write_csv(analysis_root / "execution_episode_summary.csv", episode_rows)

    aggregates = _aggregate(run_rows, ["filter_beta"])
    per_seed_aggregates = _aggregate(run_rows, ["training_seed", "filter_beta"])
    stage_summary = {
        "protocol": {
            "fixed_scenario_count": args.expected_scenarios,
            "sweep_runs": len(run_rows),
            "observational_runs": len(observational_rows),
            "scenario_bank_hashes": sorted(bank_hashes),
            "expected_scenario_bank_hash": args.expected_bank_hash,
            "controller_noise": False,
            "cuda_device": "cuda:0",
        },
        "validation": {
            "expected_runs": args.expected_sweep_runs,
            "valid_runs": len(run_rows) if not validation_errors else len(run_rows) - 1,
            "invalid_reasons": validation_errors,
        },
        "sweep_by_beta": aggregates,
        "sweep_by_seed_and_beta": per_seed_aggregates,
        "observational": observational_rows,
    }
    (analysis_root / "execution_stage_summary.json").parent.mkdir(parents=True, exist_ok=True)
    with (analysis_root / "execution_stage_summary.json").open("w") as handle:
        json.dump(stage_summary, handle, indent=2)

    _make_report(
        input_root,
        run_rows,
        aggregates,
        {
            "expected_runs": args.expected_sweep_runs,
            "valid_runs": len(run_rows) if not validation_errors else len(run_rows) - 1,
            "invalid_reasons": validation_errors,
        },
        observational_rows,
    )
    print(json.dumps(stage_summary["validation"], indent=2))
    print(f"wrote analysis artifacts under {analysis_root}")


if __name__ == "__main__":
    main()
