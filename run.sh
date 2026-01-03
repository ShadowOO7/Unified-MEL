#!/usr/bin/env bash
# =============================================================================
# Run Script for UniMEL
# Usage: ./run.sh <GPU_ID> <config_basename>
# Example: ./run.sh 0 wikidiverse
# =============================================================================

# Exit immediately if a command exits with a non-zero status,
# Treat unset variables as an error, and fail if any command in a pipe fails.
set -euo pipefail

# --- Help / Usage Check ---
if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  echo "Usage: $0 <GPU_ID> <config_basename>"
  echo "Example: $0 0 wikidiverse"
  exit 0
fi

# --- Argument Parsing ---
GPU_ID="${1-}"
CONFIG_BASENAME="${2-}"

# Verify arguments are present
if [[ -z "$GPU_ID" || -z "$CONFIG_BASENAME" ]]; then
  echo "❌ ERROR: Missing arguments."
  echo "Usage: $0 <GPU_ID> <config_basename>"
  echo "Example: $0 0 wikidiverse"
  exit 2
fi

# --- Config Validation ---
CONFIG_PATH="./config/${CONFIG_BASENAME}.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "❌ ERROR: Config file not found at: $CONFIG_PATH"
  echo "---------------------------------------------------"
  echo "Available config files in ./config/:"
  ls -1 ./config/*.yaml 2>/dev/null || echo "(No yaml files found)"
  exit 3
fi

# --- Execution ---
echo "✅ Configuration Found: $CONFIG_PATH"
echo "🚀 Starting run on Device: ${GPU_ID} | Dataset: ${CONFIG_BASENAME}"

# Set GPU visibility
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Run Python script
# -u forces unbuffered stdout/stderr (useful for logs appearing instantly)
python -u ./code/main.py --config "${CONFIG_PATH}"
