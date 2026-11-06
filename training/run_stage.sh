#!/bin/bash
# One LoRA SFT stage on the box (training/configs/sft_text.yaml), warm-started from an adapter dir (HF adapter files:
# adapter_config.json + adapter_model.safetensors, e.g. a previous round):
#   GPUS="0 1 2 3" bash training/run_stage.sh NAME DATA.jsonl INIT_ADAPTER MAX_STEPS [extra automodel overrides...]
# INIT_ADAPTER=none: a fresh adapter (no warm start). Env: MODEL (base checkpoint), CONFIG (AutoModel yaml), GBS, LBS.
# AutoModel restores model + optimizer + rng + dataloader together and has no "start from this adapter" option, so:
#   1) bootstrap: 1 step -> a checkpoint in this GPU layout (at the real lr: the restored optimizer keeps its lr, and
#      the step only moves the throwaway bootstrap adapter)
#   2) its model/ adapter <- INIT_ADAPTER (byte-checked); training/stage_init.py -> CK/NAME/epoch_0_step_0
#   3) train MAX_STEPS from it; checkpoints every 100 steps and at the end. Log: CK/NAME.log; done marker: CK/NAME/DONE
set -eu
set -m  # background jobs get their own process group: a failed launch check kills every rank
NAME=$1 DATA=$2 INIT=$3 STEPS=$4
shift 4
read -ra G <<< "${GPUS:-0 1 2 3}"
N=${#G[@]}
WORK=${WORK:-/srv/nemo}
cd "$WORK"
CK=$WORK/checkpoints
mkdir -p "$CK"
OUT=$CK/$NAME.lock && mkdir -p "$OUT"
# One run per output dir: a second launch into a live run refuses (it would clobber logs or double-use GPUs).
if [ -f "$OUT/RUNNING" ] && kill -0 "$(cat "$OUT/RUNNING")" 2>/dev/null; then
  echo "already running in $OUT (pid $(cat "$OUT/RUNNING"))" >&2; exit 1
fi
echo $$ > "$OUT/RUNNING"
trap 'rm -f "$OUT/RUNNING"' EXIT
LBS=${LBS:-1} GBS=${GBS:-16}  # the recipe: 16 rows a step, 1 per GPU (rows reach 16k tokens)
MODEL=${MODEL:-$WORK/lightning-bf16} CONFIG=${CONFIG:-training/configs/sft_text.yaml}
export PYTHONPATH=$WORK/training PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${G[*]}")
am() {
  venv-automodel/bin/automodel --nproc-per-node "$N" "$CONFIG" --distributed.ep_size "$N" \
    --model.pretrained_model_name_or_path "$MODEL" --dataset.path_or_dataset_id "$DATA" \
    --dataset.seq_length 16384 --step_scheduler.local_batch_size $LBS --step_scheduler.global_batch_size $GBS "$@"
}
[ -f "$CK/$NAME.log" ] && mv "$CK/$NAME.log" "$CK/$NAME.log.$(date +%s)"  # a relaunch (resume) keeps the earlier log
if [ "$INIT" = none ]; then
  am --checkpoint.checkpoint_dir "$CK/$NAME" --step_scheduler.max_steps "$STEPS" --step_scheduler.ckpt_every_steps "$STEPS" \
    "$@" > "$CK/$NAME.log" 2>&1 &
else
if [ ! -d "$CK/${NAME}_boot/epoch_0_step_0" ]; then
  am --checkpoint.checkpoint_dir "$CK/${NAME}_boot" --step_scheduler.max_steps 1 --step_scheduler.ckpt_every_steps 1 \
    > "$CK/${NAME}_boot.log" 2>&1
fi
if [ ! -d "$CK/$NAME/epoch_0_step_0" ]; then
  cp "$INIT"/adapter_model.safetensors "$INIT"/adapter_config.json "$CK/${NAME}_boot/epoch_0_step_0/model/"
  cmp "$INIT"/adapter_model.safetensors "$CK/${NAME}_boot/epoch_0_step_0/model/adapter_model.safetensors"
  venv-automodel/bin/python training/stage_init.py "$CK/${NAME}_boot/epoch_0_step_0" "$CK/$NAME"
fi
am --checkpoint.checkpoint_dir "$CK/$NAME" --checkpoint.restore_from epoch_0_step_0 --step_scheduler.max_steps "$STEPS" \
  --step_scheduler.ckpt_every_steps 100 "$@" > "$CK/$NAME.log" 2>&1 &
fi
# Assume the launch failed until proven otherwise: a first step, at a nonzero lr, with a rank on every requested GPU.
# (the first step logged: step 0, or the step after the checkpoint a relaunch resumed from)
for _ in $(seq 120); do grep -qE "\| step [0-9]+ \|" "$CK/$NAME.log" 2>/dev/null && break; kill -0 $! 2>/dev/null || break; sleep 10; done
grep -qE "\| step [0-9]+ \|" "$CK/$NAME.log" || { echo "launch check: no first step (see $CK/$NAME.log)" >&2; kill -- -$! 2>/dev/null; exit 1; }
for g in "${G[@]}"; do  # every requested GPU holds a rank
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -gt 10000 ] || { echo "launch check: GPU $g idle (${used} MiB)" >&2; kill -- -$!; exit 1; }
done
if grep -m1 -E "\| step [0-9]+ \|" "$CK/$NAME.log" | grep -q "lr 0.00e+00"; then echo "launch check: lr is 0" >&2; kill -- -$!; exit 1; fi
# Watchdog: a rank that dies leaves the others hung in NCCL forever; kill every rank on a traceback / OOM or a log
# that stops moving for 10 min, and fail.
while kill -0 $! 2>/dev/null; do
  sleep 30
  if grep -q -E "OutOfMemoryError|Traceback \(most recent" "$CK/$NAME.log"; then
    echo "watchdog: error in $CK/$NAME.log" >&2; kill -- -$!; exit 1
  fi
  if [ $(( $(date +%s) - $(stat -c %Y "$CK/$NAME.log") )) -gt 600 ]; then
    echo "watchdog: $CK/$NAME.log silent for 10 min" >&2; kill -- -$!; exit 1
  fi
done
wait $!
touch "$CK/$NAME/DONE"
