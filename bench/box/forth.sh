#!/bin/bash
# work/forth.sh SET OUT.json IDX TEMP  -- condition forth: -Forth composite (kernel in the weights), offline vLLM on GPU 3
cd /workspace/rpnbench
FORTH_VISIBLE=1 CUDA_VISIBLE_DEVICES=3 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_COMPILE_CACHE=1 \
PYTHONPATH=kernel/composite:kernel PROBLEMS=/workspace/rpnbench/work/$1.flat.json SYSTEM=training/data/cards/reference_v4.txt \
THINK=12000 GRACE=1000 TURN=24000 TEMP=$4 MAX_SEQS=${MAX_SEQS:-64} \
VLLM_KW="{\"compilation_config\": {\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}, \"gpu_memory_utilization\": 0.92}" \
exec venv-kernel/bin/python kernel/composite/e2e_vllm.py /workspace/rpnbench/forth-r7 $2 $3 ""
