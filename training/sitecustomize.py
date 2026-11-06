"""Loaded by every Python process with training/ on PYTHONPATH (the GRPO driver and every Ray worker it starts). When
nemo_rl.experience.rollouts is imported, its ASYNC generation step is wrapped so the Forth env (training/grpo_forth.py)
works as in the sync rollouts:
  - the generation is decoded with special tokens (the env ends a turn at <|im_end|>, sees </think>);
  - a stop at max_new_tokens is not 'truncation' (NeMo RL's async per-sample rollout ends the sample then; the sync one
    carries on): the env generates in chunks and ends turns itself (EOS, cap).
shortcut: a monkeypatch of NeMo RL's rollouts; drop it if NeMo RL grows env-controlled per-turn max_tokens.
"""
import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "nemo_rl.experience.rollouts"


def _patch(R):
    orig = R.generate_responses_async

    async def gen(policy_generation, generation_input_data, batch, tokenizer, *a, **k):
        batch, ids, m = await orig(policy_generation, generation_input_data, batch, tokenizer, *a, **k)
        for i, g in enumerate(ids):
            batch["message_log"][i][-1]["content"] = tokenizer.decode(g, skip_special_tokens=False)
        m.pop("_response_truncated", None)
        return batch, ids, m

    R.generate_responses_async = gen
    print(f"[grpo_forth sitecustomize] async rollouts: specials kept, length stops continue (pid {os.getpid()})",
          flush=True)


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("GRPO_FORTH_ASYNC_PATCH") == "1":
    sys.meta_path.insert(0, _Hook())
