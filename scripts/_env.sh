#!/usr/bin/env bash
# Standard environment prelude for evaluation pipeline.
# RUN_ID is primary; DATE is kept for backward compatibility.

: "${RUN_ID:=${DATE:-$(date -u +%Y%m%d_%H%M)}}"
DATE="$RUN_ID"
RUNS="runs/$RUN_ID"
REPORTS="reports/$RUN_ID"
STORES="$RUNS/stores"
ADAPTERS="adapters/$RUN_ID"
: "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
# Expose model path for Python scripts expecting HF_MODEL_PATH.
: "${HF_MODEL_PATH:=$MODEL}"
export MODEL HF_MODEL_PATH
# Deterministic session identifier for HEI-NW replay
HEI_SESSION_ID="hei_${RUN_ID}"
mkdir -p "$RUNS" "$REPORTS" "$STORES" "$ADAPTERS"
