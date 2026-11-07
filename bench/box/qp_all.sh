#!/bin/bash
# condition 4: Qwenputer (RISC-V release) itself, their own evalarith (rv1, greedy, budget 4096, batch 8), on GPU $1
cd /workspace/rpnbench/qp/qwenputer/experimental/riscv
export HOME=/workspace/rpnbench/qp/home XDG_CACHE_HOME=/workspace/rpnbench/cache CUDA_VISIBLE_DEVICES=$1
OUT=/workspace/rpnbench/bench/raw/qwenputer
for s in mul_big div_big add_or_sub_big mul_div_multiple_longer; do
  .venv/bin/python -m chipformer.rv.evalarith /workspace/rpnbench/qp/model --data dmx_$s --n 40 --batch 8 --out $OUT > /workspace/rpnbench/logs/qp_$s.log 2>&1
  tail -1 /workspace/rpnbench/logs/qp_$s.log
done
for d in nupa_add nupa_sub nupa_mulh nupa_mule; do
  ids=$(/workspace/rpnbench/venv-kernel/bin/python -c "import json; print(\",\".join(p[\"id\"] for p in json.load(open(\"/workspace/rpnbench/bench/sets/nupa.json\"))[\"problems\"] if p[\"id\"].startswith(\"$d-\")))")
  [ -n "$ids" ] && .venv/bin/python -m chipformer.rv.evalarith /workspace/rpnbench/qp/model --data $d --n 144 --ids $ids --batch 8 --out $OUT > /workspace/rpnbench/logs/qp_$d.log 2>&1
  tail -1 /workspace/rpnbench/logs/qp_$d.log
done
echo QPDONE
