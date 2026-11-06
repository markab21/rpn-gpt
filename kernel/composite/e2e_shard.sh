#!/bin/bash
# One e2e_demo shard (job.yml cmd): demo_set problems in LIST (a file, comma-separated, in order) through the HF composite
# on GPU, s2a_r1 + reference_v2, think 8000 / cap 16000; resumes from runs/OUT.json (disjoint LISTs per GPU, merge with
# kernel/composite/e2e_report.py). Run from a pushed tree holding forth-lightning-head/ (kernel/composite/build.py) and
# adapters/hf/stage2a_r1/.   bash kernel/composite/e2e_shard.sh GPU LIST OUT
set -e
cd "$(dirname "$0")/../.."
mkdir -p runs
export E2E_SET=training/data/demo_set.json E2E_SYS=training/data/cards/reference_v2.txt E2E_THINK=8000 E2E_TURN=16000
export PYTHONPATH=$PWD/kernel/composite CUDA_VISIBLE_DEVICES=$1
exec $PWD/venv-kernel/bin/python kernel/composite/e2e_demo.py "$PWD/forth-lightning-head" runs/$3.json $(cat "$2") adapters/hf/stage2a_r1
