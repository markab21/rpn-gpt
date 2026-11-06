"""Watch rpn-gpt answer one question live.
In the weights (default): markab/rpn-gpt-lightning-Forth served by vLLM as in playbooks/deploy.md ("The kernel in the
weights from the Hub"). A plain streamed /v1/chat/completions call: no tool, no loop. The model writes a ```forth block;
the kernel inside the same network writes the ```output block.
  python demos/live.py "Calculate the remainder when 82225043186179281 is divided by 6469193." \
      --base-url http://localhost:8000 --model forth [--gpu 0] [--svg live.svg]
External machine (--external): any /v1/completions server with a merged model, for example mlx_lm.server with
markab/rpn-gpt-lightning-MLX-4bit. This script stops the stream at each closed ```forth block, runs it in the spec
(kernel/text_vm.py) on the CPU, appends the ```output block and asks the server to go on (as rpn_chat.py does).
  python demos/live.py "..." --external --base-url http://localhost:8090 --model markab/rpn-gpt-lightning-MLX-4bit \
      --tokenizer markab/rpn-gpt-lightning-MLX-4bit --think-budget 3000
Thinking is dimmed and cut after --think-lines lines. Needs rich (and transformers for --external).
Set VLLM_API_KEY if the server has a key.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("question")
a.add_argument("--base-url", default="http://localhost:8000")
a.add_argument("--model", default="forth")
a.add_argument("--card", default=str(ROOT / "training/data/cards/reference_v4.txt"), help="the system prompt")
a.add_argument("--max-tokens", type=int, default=16000)
a.add_argument("--think-lines", type=int, default=4, help="show this many thinking lines, then only a counter")
a.add_argument("--gpu", help="show nvidia-smi for this GPU index first")
a.add_argument("--svg", help="also save the finished screen as an SVG")
a.add_argument("--external", action="store_true", help="run the Forth blocks here, in the spec on the CPU")
a.add_argument("--think-budget", type=int, default=0, help="--external: close the thinking after this many tokens")
a.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", help="--external: HF id or dir")
a.add_argument("--label", help="a line under the question, e.g. the machine it runs on")
args = a.parse_args()

URL = args.base_url.rstrip("/").removesuffix("/v1")
MSGS = [{"role": "system", "content": Path(args.card).read_text()}, {"role": "user", "content": args.question}]
OUT_TITLE = ("output: written by the Forth machine in Python on the CPU (outside the model)" if args.external else
             "output: written by the Forth kernel inside the model, no tool call")
usage = {}


def sse(path, body):
    """The JSON events of one streamed OpenAI-style request."""
    req = urllib.request.Request(URL + path, json.dumps(body | {"stream": True, "temperature": 1.0, "top_p": 0.95}).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")})
    with urllib.request.urlopen(req) as r:  # leaving early closes the stream, and the server stops generating
        for raw in r:
            raw = raw.decode().strip()
            if raw.startswith("data: ") and raw != "data: [DONE]":
                yield json.loads(raw[6:])


def in_weights():
    for ev in sse("/v1/chat/completions", {"model": args.model, "messages": MSGS, "max_tokens": args.max_tokens,
                                           "stream_options": {"include_usage": True}}):
        usage.update(ev.get("usage") or {})
        for ch in ev.get("choices", []):
            d = ch["delta"]
            if ch.get("finish_reason") and not d.get("content"):
                continue  # the closing chunk is empty too, but it is not a kernel token
            yield (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")


def external():
    sys.path.insert(0, str(ROOT / "training"))
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # no "PyTorch was not found" line: only the tokenizer is used
    import loop_sim as L
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt = tok.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True)
    text, blocks, n = "", [], 0
    while n < args.max_tokens:
        closed = False
        for ev in sse("/v1/completions", {"model": args.model, "prompt": prompt + text, "max_tokens": args.max_tokens - n}):
            piece = ev["choices"][0].get("text") or ""
            n += 1
            open_at = text.rfind("```forth\n")
            text += piece
            if open_at >= 0 and "</think>" not in text[open_at:] and "```output" not in text[open_at:] and \
                    (k := text.find("\n```", open_at + 9)) >= 0:
                blocks.append(text[open_at + 9:k].strip())  # a block closed: stop the stream here and run it
                piece, text, closed = piece[:len(piece) - (len(text) - k - 4)], text[:k + 4], True
            yield piece
            if closed:
                break
            if args.think_budget and n >= args.think_budget and "</think>" not in text and "```forth\n" not in \
                    text[text.rfind("```"):]:  # budget control as in loop_sim: close the thinking, the answer goes on
                text += "\n</think>\n\n"
                yield "\n</think>\n\n"
                break
        else:
            break  # the model ended its turn
        if closed:
            o = f"\n```output\n{L.kernel_output(blocks, tok)}\n```\n"
            text += o
            yield o
    usage["completion_tokens"] = len(tok(text, add_special_tokens=False).input_ids)


con = Console()
rec = Console(record=True, file=io.StringIO(), width=con.width, force_terminal=True)  # --svg: the prints, no status line


def show(*a, **k):
    con.print(*a, **k)
    rec.print(*a, **k)


if args.gpu is not None:
    q = "name,memory.used,memory.total,utilization.gpu"
    gpu = subprocess.run(["nvidia-smi", "-i", args.gpu, f"--query-gpu={q}", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    show(Text(f"GPU {args.gpu}: {gpu}", style="cyan"))
show(Panel(Text(args.question, style="bold"), title=f"question to {args.model}", border_style="cyan",
                subtitle=args.label))
state = {"think": True, "fence": None, "block": [], "shown": 0, "think_tok": 0, "hidden": 0, "hidden_all": 0}


def emit(ln):
    """Print one finished line where it sits: thinking, a forth block, an output block, or the answer."""
    s = state
    if "</think>" in ln:  # the chat template opens <think> in the prompt; the model closes it in the stream
        before, ln = ln.split("</think>", 1)
        emit(before)
        if s["think"]:
            show(Text(f"  (thinking: {s['think_tok']} tokens)\n", style="dim"))
        s["think"] = False
        if not ln.strip():
            return
    t = ln.strip()
    if s["fence"] is None and t in ("```forth", "```output"):
        s["fence"], s["block"] = t[3:], []
    elif s["fence"] and t == "```":
        code = "\n".join(s["block"])
        if s["fence"] == "forth":
            show(Panel(Syntax(code, "forth", theme="monokai"), title="program (written by the model)",
                            border_style="yellow", expand=False))
        else:
            show(Panel(Text(code, style="bold green"), title=OUT_TITLE, border_style="green", expand=False,
                            subtitle=f"after {s['hidden']} hidden kernel tokens" if s["hidden"] else None))
            s["hidden"] = 0
        s["fence"] = None
    elif s["fence"]:
        s["block"].append(ln)
    elif s["think"]:
        if s["shown"] < args.think_lines:
            show(Text(ln, style="dim"), no_wrap=True, overflow="ellipsis")
            s["shown"] += 1
    elif ln.strip():
        show(Text(ln, style="bold cyan" if "\\boxed" in ln else "bold"))


line, t0 = "", time.time()
with Live(Text(""), console=con, transient=True, refresh_per_second=8) as status:
    for piece in external() if args.external else in_weights():
        line += piece
        if not piece and not args.external:  # a hidden kernel token decodes to nothing (the kernel's working steps)
            state["hidden"] += 1
            state["hidden_all"] += 1
        elif state["think"]:
            state["think_tok"] += 1
        while "\n" in line:
            ln, line = line.split("\n", 1)
            emit(ln)
        dt = time.time() - t0
        if not piece and state["hidden"]:
            status.update(Text(f"  the kernel is running the program: {state['hidden']} hidden tokens, {dt:.0f} s",
                               style="green"))
        elif state["think"] and state["shown"] >= args.think_lines:
            status.update(Text(f"  thinking ... {state['think_tok']} tokens, {dt:.0f} s", style="dim"))
        else:
            status.update(Text(""))
    if line:
        emit(line)
dt = time.time() - t0
n = usage.get("completion_tokens", 0)
kern = f", {state['hidden_all']} of them hidden kernel tokens" if state["hidden_all"] else ""
show(Text(f"\n--- {args.model} | {n} tokens{kern} | {dt:.1f} s, {n / dt:.0f} tok/s", style="dim"))
if args.svg:
    rec.save_svg(args.svg, title="rpn-gpt")
