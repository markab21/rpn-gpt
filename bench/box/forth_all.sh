#!/bin/bash
# all condition-3 runs, one engine start per (set, kind); a finished file is skipped (resume)
cd /workspace/rpnbench
source work/env.sh  # exports VLLM_API_KEY, LIGHTNING_URL (the stock server), LOOP_MODEL=lightning, LOOP_TOKENIZER, cache dirs
export VLLM_NO_USAGE_STATS=1 MAX_SEQS=160
ALL=$(seq -s, 0 39); X4="$ALL,$ALL,$ALL,$ALL"
for kind in sampled greedy; do
  for s in nupa mul_big div_big add_or_sub_big mul_div_multiple_longer; do
    out=bench/raw/forth/$s.$kind.json
    [ -f $out ] && continue
    if [ $kind = sampled ]; then idx=$X4; t=1.0; else idx=$ALL; t=0; fi
    echo "=== $s $kind $(date)"
    work/forth.sh $s $out.tmp $idx $t > logs/forth_$s.$kind.log 2>&1 && mv $out.tmp $out
    tail -1 logs/forth_$s.$kind.log
  done
done
echo ALLDONE
