#!/bin/bash
# Full SANA experiment pipeline.
# Usage:
#   ./run_sana_all.sh                          # default: gpu=0, model=sana1.5_1.6b
#   ./run_sana_all.sh -g 3                     # specify GPU
#   ./run_sana_all.sh -g 3 -m sana1.5_4.8b    # specify GPU and model
#   ./run_sana_all.sh -g 3 -m sana1.5_4.8b --desc my_run

set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
GPU="0"
MODEL="sana1.5_1.6b"
DESC="camera_ready"
TOTAL_STEPS=500
INSTANCES=""   # empty = all instances

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)       GPU="$2";         shift 2 ;;
    -m|--model)     MODEL="$2";       shift 2 ;;
    --desc)         DESC="$2";        shift 2 ;;
    --steps)        TOTAL_STEPS="$2"; shift 2 ;;
    --instances)
      shift
      INSTANCES=""
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        INSTANCES="$INSTANCES $1"
        shift
      done
      ;;
    -h|--help)
      echo "Usage: $0 [-g GPU] [-m MODEL] [--desc DESC] [--steps N] [--instances i1 i2 ...]"
      echo ""
      echo "  -g, --gpu        GPU id (default: 0)"
      echo "  -m, --model      sana1.5_1.6b | sana1.5_4.8b (default: sana1.5_1.6b)"
      echo "  --desc           Experiment description suffix (default: camera_ready)"
      echo "  --steps          Total training steps (default: 1000)"
      echo "  --instances      Specific instances to train/eval (default: all)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

INST_ARGS=""
if [[ -n "$INSTANCES" ]]; then
  INST_ARGS="--instances $INSTANCES"
fi

echo "============================================"
echo " SANA Experiment Pipeline"
echo " GPU:       $GPU"
echo " Model:     $MODEL"
echo " Desc:      $DESC"
echo " Steps:     $TOTAL_STEPS"
echo " Instances: ${INSTANCES:-all}"
echo "============================================"

# ── 1. Ours (DTI) + eval ────────────────────────────────────────────────────
echo ""
echo ">>> [1/4] Running Ours (DTI) ..."
uv run python exps/ours_sana.py \
  -g "$GPU" \
  -m "$MODEL" \
  --total_steps "$TOTAL_STEPS" \
  --desc "$DESC" \
  $INST_ARGS

# ── 2. TI + eval + eval(TI-rescaled to mean) ────────────────────────────────
echo ""
echo ">>> [2/4] Running Textual Inversion (TI) ..."
uv run python exps/ti_sana.py \
  -g "$GPU" \
  -m "$MODEL" \
  --total_steps "$TOTAL_STEPS" \
  --desc "$DESC" \
  $INST_ARGS

# Re-evaluate TI with embeddings rescaled to mean vocab norm.
TI_EXPDIR="outputs/ti-${MODEL}"
if [[ -n "$DESC" ]]; then
  TI_EXPDIR="${TI_EXPDIR}-${DESC}"
fi
echo ""
echo ">>> [2b/4] Re-evaluating TI with --rescale mean ..."
CMD="uv run python scripts/evaluate.py \
  -g $GPU \
  -e=$TI_EXPDIR \
  --checkpoint=$TOTAL_STEPS \
  --rescale=mean \
  --out_dir=images_rescaled"
if [[ -n "$INSTANCES" ]]; then
  CMD="$CMD --instances $INSTANCES"
fi
echo "  eval checkpoint $TOTAL_STEPS (rescaled) ..."
eval $CMD

# ── 3. CrossInit + eval ─────────────────────────────────────────────────────
echo ""
echo ">>> [3/4] Running CrossInit ..."
# Use ours_sana.py with xinit init_method (no dedicated xinit_sana.py).
uv run python exps/ours_sana.py \
  -g "$GPU" \
  -m "$MODEL" \
  --total_steps "$TOTAL_STEPS" \
  --desc "xinit-${DESC}" \
  --init_method xinit \
  --kappa 0.0 \
  $INST_ARGS

# ── 4. Ours (DTI) — infinite loop ──────────────────────────────────────────
echo ""
echo ">>> [4/4] Running Ours (DTI) in infinite loop (Ctrl+C to stop) ..."

child_pid=0
cleanup() {
  echo ""
  echo "Received stop signal, terminating child..."
  if [[ "$child_pid" -ne 0 ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
  echo "  Starting DTI experiment: $(date)"
  uv run python exps/ours_sana.py \
    -g "$GPU" \
    -m "$MODEL" \
    --total_steps "$TOTAL_STEPS" \
    --desc "$DESC" \
    $INST_ARGS &
  child_pid=$!
  wait "$child_pid"
  status=$?
  echo "  Process exited with status $status. Restarting in 5 seconds..."
  child_pid=0
  sleep 5
done
