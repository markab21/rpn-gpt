"""Upload files to the private HF dataset (or, with --model, the adapter model repo) in ONE commit and print the commit
id to pin (stage.yml data_rev).
  python training/hf_put.py --msg "distill r1 rows" lora_runs/r1_rows.jsonl=distill/r1/rows.jsonl [more local=remote ...]
Needs a write token in HF_TOKEN and your repo names in RPN_HF_DATASET / RPN_HF_MODEL.
"""

import argparse
import os

from huggingface_hub import CommitOperationAdd, HfApi

DATASET, MODEL = os.environ.get("RPN_HF_DATASET", "<you>/rpn-gpt-data"), os.environ.get("RPN_HF_MODEL", "<you>/rpn-gpt-lora")


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("files", nargs="+", help="local=path/in/repo")
    a.add_argument("--msg", required=True)
    a.add_argument("--model", action="store_true", help=f"commit to {MODEL} (adapters) instead of the dataset")
    args = a.parse_args()
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for l, r in (f.split("=", 1) for f in args.files)]
    repo, kind = (MODEL, "model") if args.model else (DATASET, "dataset")
    info = HfApi().create_commit(repo, ops, commit_message=args.msg, repo_type=kind)
    print(f"commit {info.oid} ({len(ops)} files): {info.commit_url}")


if __name__ == "__main__":
    main()
