#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SCENARIO_BANK="datasets/phase3/scenario_bank_512.pt"
ESTIMATOR="motion_estimator/artifacts/20260909_162237_v1_1_C/best.pt"
OUTPUT_ROOT="training/legged_gym/logs/go2_pos_dynamic/phase3_repeated_eval"
ANALYSIS_DIR="training/legged_gym/logs/go2_pos_dynamic/phase3B_analysis"

echo "[phase3B] repository=${SCRIPT_DIR}"
echo "[phase3B] CUDA_VISIBLE_DEVICES=0"
echo "[phase3B] output_root=${OUTPUT_ROOT}"

# The Python runner executes each (training seed, method, repeat) in a fresh
# process, in a fixed serial order.  It skips only directories containing all
# three required artifacts, so the command is safe to resume from tmux after
# an interruption.
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n sea_nav python \
  training/legged_gym/legged_gym/scripts/run_repeated_evaluations.py \
  --scenario_bank "${SCENARIO_BANK}" \
  --estimator_checkpoint "${ESTIMATOR}" \
  --output_root "${OUTPUT_ROOT}" \
  --num_envs 64 \
  --max_steps_per_episode 3000 \
  --monitor_envs 0 \
  --progress_interval_steps 500 \
  --model 1 A training/legged_gym/logs/go2_pos_dynamic/09_11_11-36-25_phase2_full_A_original_seed1/model_2000.pt \
  --model 1 B training/legged_gym/logs/go2_pos_dynamic/09_11_13-07-26_phase2_full_B_sync_static_seed1/model_2000.pt \
  --model 1 C training/legged_gym/logs/go2_pos_dynamic/09_11_14-38-45_phase2_full_C_predictive_seed1/model_2000.pt \
  --model 2 A training/legged_gym/logs/go2_pos_dynamic/09_12_13-47-47_phase3_seed2_A/model_2000.pt \
  --model 2 B training/legged_gym/logs/go2_pos_dynamic/09_12_17-27-48_phase3_seed2_B/model_2000.pt \
  --model 2 C training/legged_gym/logs/go2_pos_dynamic/09_12_21-07-10_phase3_seed2_C/model_2000.pt \
  --model 3 A training/legged_gym/logs/go2_pos_dynamic/09_12_13-48-49_phase3_seed3_A/model_2000.pt \
  --model 3 B training/legged_gym/logs/go2_pos_dynamic/09_12_17-28-50_phase3_seed3_B/model_2000.pt \
  --model 3 C training/legged_gym/logs/go2_pos_dynamic/09_12_21-08-11_phase3_seed3_C/model_2000.pt

# This stage is CPU-only and never starts another simulator process.
conda run --no-capture-output -n sea_nav python \
  training/legged_gym/legged_gym/scripts/aggregate_repeated_evaluations.py \
  --root "${OUTPUT_ROOT}" \
  --output_dir "${ANALYSIS_DIR}" \
  --seeds 1 2 3 \
  --methods A B C \
  --repeats 1 2 3

echo "[phase3B] evaluation and aggregation complete"
