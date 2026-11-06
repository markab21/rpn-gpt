"""Ask rpn-gpt one question on any server, with the Forth machine: the model writes ```forth blocks; this loop runs
each block in the spec (kernel/text_vm.py), appends the ```output block, and lets the model go on. Prints the turn.
  uv run python rpn_chat.py "What is 7^77 mod 1000003?" --base-url http://localhost:8000 --model rpn
  --server vllm (default: vLLM, SGLang, any OpenAI /v1/completions server that honours include_stop_str_in_output)
         | llamacpp (llama-server) | ollama | mlx (mlx_lm.server)
Auth (vLLM --api-key): set VLLM_API_KEY. The prompt is built here with the tokenizer's chat template (--tokenizer).
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training"))
import loop_sim as L  # noqa: E402

a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("question")
a.add_argument("--base-url", default="http://localhost:8000", help="server root (a trailing /v1 is fine)")
a.add_argument("--model", default="rpn", help="the served model name (vLLM --served-model-name / LoRA name, Ollama tag)")
a.add_argument("--server", default="vllm", choices=["vllm", "llamacpp", "ollama", "mlx"])
a.add_argument("--card", default=str(ROOT / "training/data/cards/reference_v4.txt"), help="the system prompt")
a.add_argument("--think-budget", type=int, default=12000, help="close the thinking after this many tokens")
a.add_argument("--max-tokens", type=int, default=24000, help="the whole turn's budget")
a.add_argument("--no-think", action="store_true")
a.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", help="HF id or local dir")
args = a.parse_args()

base = args.base_url.rstrip("/").removesuffix("/v1")
L.SERVER, L.MODEL = args.server, args.model
L.URLS = [base + {"vllm": "/v1/completions", "mlx": "/v1/completions", "llamacpp": "/completion",
                  "ollama": "/api/generate"}[args.server]]
L.sp.MAX_TOKENS = 10**6
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.tokenizer)
with contextlib.redirect_stdout(io.StringIO()):  # solve() prints a grading line; there is no expected answer here
    r = L.solve({"q": args.question, "a": ""}, Path(args.card).read_text(), tok, tok, not args.no_think, 12,
                args.max_tokens, args.think_budget)
print(r["text"])
print(f"\n--- {len(r['blocks'])} Forth block(s) run, {r['tokens']} tokens, answer: {r['got']}")
