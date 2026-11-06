# Playbooks

This page shows each step of the project, from an empty GPU machine to a measured adapter. The GPU steps are Ansible playbooks in [`ansible/`](ansible/), run from your laptop. The CPU steps run on the laptop itself.

To run the released model on one GPU or a Mac without Ansible, follow [Deploy and test](deploy.md).

The steps use [uv](https://docs.astral.sh/uv/), [Ansible](https://docs.ansible.com/), [vLLM](https://docs.vllm.ai/) 0.29, [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) (LoRA SFT), [NeMo RL](https://github.com/NVIDIA-NeMo/RL) (its lock file builds the environments, and GRPO), and the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub).

## On your laptop

The spec, the Forth helpers, the problem generators, the row builders and the grader need no GPU.

```sh
uv sync
uv run python kernel/text_vm.py                        # the spec's self-check
uv run python -m forth.run ': sq dup * ; 12 sq .'      # run Forth through the spec
uv run python kernel/show_stream.py '2 3 + .'          # one block's token stream
uv run --group dev pytest -q                           # spec, grader and every doc example
```

The first run downloads the pinned Lightning tokenizer (a few MB). For a few rollouts on a Mac, [`training/mlx_local.sh`](../training/mlx_local.sh) serves Lightning 8-bit with MLX. Keep these local runs to a few problems.

## Set up a GPU machine

We used the playbooks on a Linux machine with 8 RTX PRO 6000 GPUs (sm_120) and an NGC PyTorch container. For other GPUs, change `build_env` in [`site.yml`](ansible/site.yml).

1. Copy [`ansible/inventory.example.yml`](ansible/inventory.example.yml) to `ansible/inventory.yml` and fill in your machine's address.
2. If you want to push rows and adapters to the Hub, export `HF_WRITE_TOKEN` in your shell first. The playbook writes it to the machine's `/etc/environment` and never prints it.
3. Run the setup:

```sh
ansible-playbook -i playbooks/ansible/inventory.yml playbooks/ansible/site.yml
```

The setup copies this repo to `/srv/nemo` on the machine (change it with `-e work=/path`). It downloads the pinned Lightning BF16 and NVFP4 checkpoints. It builds three uv environments from NeMo RL's lock: `venv-kernel` (vLLM 0.29 and transformers 5.16.1, for the kernel and serving), `venv-automodel` (training) and `venv-rl-driver` (GRPO). It also makes a random `VLLM_API_KEY`. A second run changes nothing on a ready machine. `--tags verify` only checks.

Every playbook treats a launch as failed until the machine shows otherwise: the GPUs are free before, the GPUs hold the job after, and the job is still alive ([`tasks/launched.yml`](ansible/tasks/launched.yml)).

## Build the kernel and test parity

```sh
ansible-playbook -i playbooks/ansible/inventory.yml playbooks/ansible/kernel_test.yml -e gpu=0 \
  [-e progs=kernel/tests/kernel_turns_2d.json] [-e tag=mytest]
```

[`kernel/sp_test.py`](../kernel/sp_test.py) builds the kernel with [`sp_kernel.py`](../kernel/sp_kernel.py), saves it, and reloads it with `from_pretrained`. Then it checks every kernel-owned position of every case against the spec, in prefill and in cached decode. It passes only if every case is exact. The result JSON comes back to `kernel/results/`.

Before you change the spec, measure the change on real programs (CPU):

```sh
uv run python kernel/replay.py ROLLOUTS.json            # outputs and answers that change
uv run python kernel/code_measure.py a1 OUT.jsonl ROLLOUTS.json   # code-memory variants
```

## Build the composite

```sh
ansible-playbook -i playbooks/ansible/inventory.yml playbooks/ansible/composite.yml
```

[`kernel/composite/build.py`](../kernel/composite/build.py) writes `forth-lightning/`: the composite config, the kernel weights, a few override rows and the modeling code. The base is referenced by path and never copied. Then prove it in vLLM with [`vllm_kernel_check.py`](../kernel/composite/vllm_kernel_check.py) through `job.yml` (below).

## Serve a model

For rollouts with the external machine, serve plain Lightning. The spec computes the output blocks in the rollout loop.

```sh
ansible-playbook -i playbooks/ansible/inventory.yml playbooks/ansible/serve.yml \
  -e model=lightning-nvfp4 -e gpu=1 -e port=8201 [-e "loras='s2a_r5=/srv/nemo/adapters/s2a_r5'"]
```

`model=lightning` serves BF16. With `loras`, vLLM serves each adapter under its name. Adapters train on BF16 and serve on the NVFP4 base.

To serve the composite (the kernel in the weights) with an adapter, run [`kernel/composite/serve_forth.sh`](../kernel/composite/serve_forth.sh) through `job.yml`:

```sh
ansible-playbook -i playbooks/ansible/inventory.yml playbooks/ansible/job.yml -e tag=forth_srv -e gpus=2 \
  -e "cmd='bash kernel/composite/serve_forth.sh /srv/nemo/adapters/s2a_r5'"
```

## Generate problems

```sh
LOOP_TOKENIZER=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 uv run python training/mass_sets.py
uv run python training/dev_round.py 407
```

[`mass_sets.py`](../training/mass_sets.py) carves the frozen test out first, then writes the sampling pool. The other set builders are listed in [training](../training/README.md#files). Measure a no-machine baseline on any new set before you train on it.

## Roll out

[`training/loop_sim.py`](../training/loop_sim.py) runs full turns against a server, with the spec as the external machine. It appends each rollout to `OUT.json.partial.jsonl`, so the same command resumes a stopped run.

```sh
CMD="LOOP_MODEL=s2a_r5 LIGHTNING_URL=http://localhost:8201/v1/completions venv-kernel/bin/python training/loop_sim.py training/data/dev_round.json training/data/cards/reference_v4.txt --think --think-budget 12000 --max-tokens 24000 --samples 4 --out runs/r5_dev.json"
ansible-playbook -i playbooks/ansible/inventory.yml playbooks/ansible/job.yml -e tag=r5_dev -e gpus=none -e "cmd='$CMD'"
```

Use think 12000, turn 24000, and the model card's sampling (temperature 1.0, top-p 0.95). Run 1 to 10 problems first and read them.

## Grade

The exact grader runs inside `loop_sim.py`. To regrade after a grader change, or to grade MATH-style answers:

```sh
uv run --with math_verify python training/math_grade.py runs/math500.json runs/math500.graded.json
uv run python training/mass_report.py PROBLEMS.json OUT_PREFIX runs/*.json   # pass rates per family and kind
```

Before the next round, read the wrong rollouts and sort them into buckets: token cap, by hand, re-sent block, wrong program, machine limit, format, grader. A grader false negative looks like a skill gap.

## Build rows

```sh
uv run python training/r3_rows.py --pool POOL.json --src runs/A.json:training/data/cards/reference_v4.txt:12000 \
  --prompt training/data/cards/reference_v4.txt --out rows/ --build
LOOP_TOKENIZER=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  uv run python training/r5_demos.py --prompt training/data/cards/reference_v4.txt --out demos/
```

The builders keep a row only if it is correct, clean, and exact against the composite's token stream. Upload the rows to your dataset in one commit and note the commit id:

```sh
RPN_HF_DATASET=you/your-data uv run python training/hf_put.py --msg "round 5 rows" rows/train.jsonl=stage2a/r5/rows/train.jsonl
```

## Train a LoRA round

```sh
ansible-playbook -i playbooks/ansible/inventory.yml playbooks/ansible/stage.yml \
  -e name=s2a_r5 -e data_repo=you/your-data -e data=stage2a/r5/rows/train.jsonl -e data_rev=COMMIT \
  -e init=none -e steps=465 -e gpus=0,1 [-e val=stage2a/r5/rows/val.jsonl]
```

[`training/run_stage.sh`](../training/run_stage.sh) runs NeMo AutoModel with [`training/configs/sft_text.yaml`](../training/configs/sft_text.yaml) (LoRA rank 16, learning rate 1e-4, 16 rows a step). It checks the launch (a first step, a nonzero learning rate, a rank on every GPU). Its watchdog stops every rank on a traceback, an out-of-memory error, or a log silent for 10 minutes. Train a fresh LoRA on the previous rows plus the new ones, for one epoch.

## Evaluate

The rules ([why](../training/README.md#how-we-measure)):

1. Decide each round on the dev set (`dev_round.json`, x4). Use the frozen test (x2) only at milestones.
2. Serve both adapters on the same `--enable-lora` engines and interleave their requests. Compare with a zero adapter ([`zero_adapter.py`](../training/zero_adapter.py)), never with the base model name.
3. Use one card, one budget and one grader for both arms. Regrade both arms after a grader change.
4. Run the plain-math guard for every candidate: MATH-500 x2, no machine (`training/data/cards/nomachine.txt`), against stock Lightning on the same path.

```sh
uv run python training/paired_delta.py runs/A.json runs/B.json training/data/dev_round.json
```

This prints the paired difference per kind with a 95% bootstrap interval.

## Run in the weights

Here the composite answers on its own: the kernel in the weights writes every output block, and the script checks each kernel position against the spec. Run this on the GPU machine from `/srv/nemo`, or save it as `runs/inw.sh` and run it through `job.yml` with `cmd='bash runs/inw.sh'`:

```sh
FORTH_VISIBLE=1 CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_COMPILE_CACHE=1 \
PYTHONPATH=kernel/composite:kernel PROBLEMS=training/data/demo10.json SYSTEM=training/data/cards/reference_v4.txt \
VLLM_KW='{"moe_backend": "marlin", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}' \
  venv-kernel/bin/python kernel/composite/e2e_vllm.py forth-lightning-nvfp4 runs/inw.json all /srv/nemo/adapters/s2a_r5
```

[`e2e_vllm.py`](../kernel/composite/e2e_vllm.py) runs all problems at once in one vLLM engine. [`e2e_demo.py`](../kernel/composite/e2e_demo.py) is the slower HF reference, one problem at a time. Per problem, the output gives the answer, whether every kernel position was exact, and the hidden and visible token counts. `serve_forth.sh` makes the `forth-lightning-nvfp4` directory on its first run.

## Stop a job, see the machine

```sh
ansible-playbook -i playbooks/ansible/inventory.yml playbooks/ansible/status.yml
ansible-playbook -i playbooks/ansible/inventory.yml playbooks/ansible/stop.yml -e dir=runs/job_TAG
```

`status.yml` shows each GPU (memory, use, the job that holds it) and each job's last progress line. `stop.yml` stops one job by the pid in its `RUNNING` file, children first, and waits until its GPUs are free.
