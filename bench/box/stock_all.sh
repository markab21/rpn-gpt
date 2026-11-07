#!/bin/bash
# work/stock_all.sh COND: 4 sampled runs at once, then greedy (resumable: run_stock.py skips finished rollouts)
cd /workspace/rpnbench && source work/env.sh  # exports VLLM_API_KEY, LIGHTNING_URL (the stock server), LOOP_MODEL=lightning, LOOP_TOKENIZER, cache dirs
for r in s0 s1 s2 s3; do venv-kernel/bin/python bench/run_stock.py $1 $r --workers 24 > logs/$1.$r.log 2>&1 & done
wait
LOOP_TEMP=0 venv-kernel/bin/python bench/run_stock.py $1 greedy --workers 40 > logs/$1.greedy.log 2>&1
echo DONE $1
