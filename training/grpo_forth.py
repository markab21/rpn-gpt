"""GRPO (NeMo RL) on the text-mode Forth machine, rollouts exactly as training/loop_sim.py: the model writes; when it closes
a ```forth block the machine (kernel/text_vm.py from a PINNED tree, env.forth.spec_dir) runs every block of the turn so
far and its ```output block is appended (environment tokens: no loss), generation continues. Thinking is closed with
"\\n</think>\\n\\n" at think_budget visible tokens (+GRACE while a block is open); the turn ends at <|im_end|> or at cap
visible tokens. Reward = correct(last \\boxed after </think>, loop_sim.same) - length_cost * visible tokens.
  (from /srv/nemo/RL) PYTHONPATH=<this dir>/.. uv run --extra vllm python <this file> --config CONFIG [overrides]
Generation runs in chunks of policy.generation.max_new_tokens (NeMo RL has one max_tokens per batch), so the think
budget and the cap can overshoot by up to one chunk. shortcut: chunked budget, exact per-sample limits need a NeMo RL
patch; upgrade if the overshoot shows in the token stats.
"""

import json
import os
import sys
import time

GRACE = 1000  # loop_sim.GRACE
EOS = "<|im_end|>"
CLOSE = "\n</think>\n\n"
STOPS = ["```\n", "```"]  # loop_sim.STOPS


try:  # (the CPU test imports this file without NeMo RL)
    from nemo_rl.utils.logger import LoggerInterface
except ImportError:
    LoggerInterface = object


class JsonlLogger(LoggerInterface):
    """logger.extra_backends: every metric of every step as one JSON line in $GRPO_METRICS (no wandb / tensorboard)."""
    def log_metrics(self, metrics, step, prefix="", step_metric=None, step_finished=False):
        with open(os.environ["GRPO_METRICS"], "a") as f:
            f.write(json.dumps({"step": step, "prefix": prefix, **{k: v for k, v in metrics.items()
                                if isinstance(v, (int, float, str, bool))}}) + "\n")

    def log_hyperparams(self, params):
        pass

    def log_histogram(self, histogram, step, name):
        pass

    def log_plot(self, figure, step, name):
        pass


try:
    from transformers import PreTrainedTokenizerFast as _Fast

    class KeepSpecialsTokenizer(_Fast):
        """The env must see <|im_end|> (end of turn) and </think>, but NeMo RL decodes generations with
        skip_special_tokens=True (sync rollouts; the async path: training/sitecustomize.py)."""
        def batch_decode(self, sequences, **kw):
            return super().batch_decode(sequences, **dict(kw, skip_special_tokens=False))

except ImportError:
    pass


def new_state(p, cfg):
    return {"q": p["q"], "a": str(p["a"]), "family": p.get("family"), "text": "", "blocks": [], "log": [],
            "think_budget": cfg["think_budget"], "cap": cfg["cap"], "rounds": cfg["rounds"]}


def advance(s, piece, tok, L):
    """One generation chunk `piece` (decoded WITH special tokens) -> (observation, reward, done). L = loop_sim."""
    eos = piece.endswith(EOS)
    s["text"] += piece[:-len(EOS)] if eos else piece
    text, obs = s["text"], ""
    end = text.rstrip("\n")
    if not eos and end.endswith("```") and text[len(end):] in ("", "\n", "\n\n"):  # stopped at a fence (loop_sim STOPS)
        opener = text.rfind("```forth\n")
        body = end[opener + len("```forth\n"):-3] if opener >= 0 else None
        if body is not None and "```" not in body and "</think>" not in body and len(s["log"]) <= s["rounds"]:
            s["blocks"].append(body.strip())
            out = L.kernel_output(s["blocks"], tok)
            obs = ("" if text.endswith("\n") else "\n") + f"```output\n{out}\n```\n"  # = loop_sim: end + "\n" + o + "\n"
            s["log"].append({"program": body.strip(), "output": out})
    used = len(tok(text + obs, add_special_tokens=False).input_ids)
    if eos or used >= s["cap"]:
        s["text"] += obs
        got = L.last_boxed(text.rsplit("</think>", 1)[-1])
        s.update(got=got, ok=L.same(got, s["a"], s), tokens=used, done=True)
        return obs, None, True
    if s["think_budget"] and "</think>" not in text + obs:
        k = text.rfind("```forth\n")
        budget = s["think_budget"] + GRACE * (k >= 0 and "```" not in text[k + 9:])
        if used >= budget:
            obs += CLOSE
    s["text"] += obs
    return obs, None, False


def reward(s, length_cost):
    return float(s["ok"]) - length_cost * s["tokens"]


def load_loop_sim(spec_dir):
    """loop_sim + text_vm from the pinned tree (a git archive), never a working tree."""
    sys.path[:0] = [spec_dir, os.path.join(spec_dir, "training"), os.path.join(spec_dir, "kernel")]
    import loop_sim
    assert os.path.dirname(os.path.abspath(loop_sim.__file__)) == os.path.join(os.path.abspath(spec_dir), "training"), \
        loop_sim.__file__
    loop_sim.sp.MAX_TOKENS = 10**6  # as loop_sim.main
    return loop_sim


def make_env_class():
    import ray
    import torch
    from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

    @ray.remote(num_cpus=1)
    class ForthTextEnv(EnvironmentInterface):
        def __init__(self, cfg):
            from transformers import AutoTokenizer
            self.cfg = cfg
            self.L = load_loop_sim(cfg["spec_dir"])
            self.tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
            self.log, self.done = cfg.get("log_path"), []  # finished episodes since the last metrics call
            if self.log:
                os.makedirs(os.path.dirname(self.log), exist_ok=True)

        def step(self, message_log_batch, metadata):
            obs, rewards, dones, meta = [], [], [], []
            for msgs, s in zip(message_log_batch, metadata):
                o, _, done = advance(s, msgs[-1]["content"], self.tok, self.L)
                r = reward(s, self.cfg["length_cost"]) if done else 0.0
                if done:
                    self.done.append(s)
                if done and self.log:
                    with open(self.log, "a") as f:
                        f.write(json.dumps({"t": time.time(), "q": s["q"], "family": s["family"], "a": s["a"],
                                            "got": s["got"], "ok": s["ok"], "tokens": s["tokens"], "reward": r,
                                            "blocks": s["log"], "text": s["text"]}) + "\n")
                obs.append({"role": "environment", "content": o})
                rewards.append(r)
                dones.append(done)
                meta.append(s)
            # every turn needs the fences again: None here means NO stop strings on the next turn (rollouts.py)
            return EnvironmentReturn(observations=obs, metadata=meta, next_stop_strings=[STOPS] * len(obs),
                                     rewards=torch.tensor(rewards, dtype=torch.float32),
                                     terminateds=torch.tensor(dones, dtype=torch.bool),
                                     answers=[s.get("got") for s in meta])

        def global_post_process_and_metrics(self, batch):
            # the driver keeps no metadata of finished samples (rollouts.py updates continuing ones only): count here
            done, self.done, n = self.done, [], max(len(batch["idx"]), 1)
            k = max(len(done), 1)
            return batch, {"forth_correct": sum(s["ok"] for s in done) / n, "forth_finished": len(done) / n,
                           "forth_blocks": sum(len(s["log"]) for s in done) / k,
                           "forth_uses_machine": sum(bool(s["log"]) for s in done) / k,
                           "forth_visible_tokens": sum(s["tokens"] for s in done) / k}

        def shutdown(self):
            pass

    return ForthTextEnv


def datums(problems, system, tok, cfg, task):
    import torch  # noqa: F401
    out = []
    for i, p in enumerate(problems):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": p["q"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)  # = loop_sim
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        out.append({"message_log": [{"role": "user", "content": prompt, "token_ids": ids}], "length": len(ids),
                    "extra_env_info": new_state(p, cfg), "loss_multiplier": 1.0, "idx": i, "task_name": task,
                    "stop_strings": STOPS})
    return out


def main():
    import argparse
    import pprint
    from omegaconf import OmegaConf
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
    from nemo_rl.algorithms.utils import get_tokenizer, set_seed
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
    from nemo_rl.utils.logger import get_next_experiment_dir

    register_omegaconf_resolvers()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, overrides = ap.parse_known_args()
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    pprint.pprint(config)
    config.logger.log_dir = get_next_experiment_dir(config.logger.log_dir)
    os.makedirs(config.logger.log_dir, exist_ok=True)
    os.environ["GRPO_METRICS"] = os.path.join(config.logger.log_dir, "metrics.jsonl")  # (before Ray starts: inherited)
    init_ray()
    set_seed(config.grpo.seed)
    tok = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(config.policy["generation"], tok)
    import grpo_forth  # the class from the importable module, not __main__ (the async collector unpickles it)
    assert type(tok) is grpo_forth._Fast, type(tok)
    tok.__class__ = grpo_forth.KeepSpecialsTokenizer

    fc = dict(config.env["forth"])
    system = open(fc["system_prompt"]).read()
    Env = make_env_class()
    train_env = Env.options(num_gpus=0).remote(dict(fc, log_path=os.path.join(config.logger.log_dir, "train_env.jsonl")))
    val_env = Env.options(num_gpus=0).remote(dict(fc, length_cost=0.0,  # val accuracy = plain correctness
                                                  log_path=os.path.join(config.logger.log_dir, "val_env.jsonl")))
    train = datums(json.load(open(fc["train"])), system, tok, fc, "forth")
    val = datums(json.load(open(fc["val"])), system, tok, fc, "forth") if fc.get("val") else None
    print(f"forth: {len(train)} train / {len(val or [])} val problems; prompt tokens {train[0]['length']}; "
          f"first prompt ends {tok.decode(train[0]['message_log'][0]['token_ids'][-12:])!r}", flush=True)

    (policy, policy_generation, _gym, cluster, dataloader, val_dataloader, loss_fn, logger, checkpointer, grpo_state,
     master_config, _t, _a) = setup(config, tok, train, val)
    grpo_train(policy, policy_generation, dataloader, val_dataloader, tok, loss_fn, {"forth": train_env},
               {"forth": val_env}, logger, checkpointer, grpo_state, master_config)


if __name__ == "__main__":
    main()
