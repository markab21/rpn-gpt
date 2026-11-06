"""Expert-iteration rows: train the student only where it fails, on verified kernel-loop transcripts - its OWN
successes first (on-policy), the teacher's only for problems the student never solves.
From training/loop_sim.py results on the same pool (several samples per problem), optionally Mammoth-scored
(training/rubric.py: the .rubric.json files):
  targets  = problems the student did not solve every time
  accepted = transcripts whose reasoning closed, whose ```output blocks were all written by the kernel (none typed by
             the model), and whose final answer came from a machine output: either exactly correct, or (rescue) judged
             correct by Mammoth while a machine output is exactly the expected answer (a format-only miss)
Per target, the K best accepted transcripts (rubric quality: machine-backed, tool early, concise, recovered; then
shortest) become rows: Lightning's chat-template prompt (the same system prompt + shots as the loop test) + the whole
turn + <|im_end|>, tokenized, with train_mask = 1 on what the model writes (reasoning, programs, retries, answer) and 0
on the prompt and on every kernel ```output block.
  python training/distill_rows.py --student S.json --teacher T.json --rubric lora_runs/rubric_v10.json --out rows.jsonl
Context distillation (stage 1): --rubric may be a plain-text system prompt (the SHORT prompt the rows are rebuilt with,
not the card they were generated with), --all makes every problem a target (the short-prompt model is the one that
fails), --direct accepts no-machine rollouts instead: correct, reasoning closed, no ```forth block at all, and --clean
CARD drops transcripts that copy the generating prompt CARD where the row prompt lacks it: any 8-word run of CARD that
is not in the row prompt or the question (outside the machine's output blocks; Lightning quotes whatever system prompt it
gets, so quotes of the row prompt itself are grounded and kept); also two </think>, a placeholder block ("..."), or an
identical block re-run within the thinking or within the answer, or a value before `variable` ("5 variable x !"). Also dropped: meta-talk
that credits instructions for using the machine ("as instructed", "the user wants me to use the machine").
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_sim  # noqa: E402  (loop_sim.T: the text_vm of this tree)
from loop_sim import norm  # noqa: E402
from rubric import quality  # noqa: E402

F = "`" * 3
OUT_SPAN = re.compile(F + r"output\n.*?\n" + F + r"\n", re.S)  # the harness's (= the kernel's) output blocks
ERR_SPAN = re.compile(F + r"forth\n(?:(?!" + F + r").)*\n" + F + r"\n(?=" + F + r"output\nerror)", re.S)  # a failed block


def accepted(r):
    if "</think>" not in r["text"] or any(b["program"] == "" for b in r["blocks"]):
        return False
    if r["text"].count(F + "output\n") != len(r["blocks"]):  # an output block the model typed itself
        return False
    outs = [" ".join(b["output"].split()) for b in r["blocks"]]
    if r["ok"]:
        got = norm(r["got"] or "")
        return any(got in o for o in outs)  # the answer came from the machine
    # rescue a format-only miss: the machine printed exactly the answer and the judge accepts the final answer
    return bool(r.get("judge_ok")) and norm(r["a"]) in outs  # (typed outputs were rejected above)


def grams(text, n=8):
    w = re.findall(r"\w+|[^\w\s]", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def words(s):
    return " ".join(re.findall(r"\w+", s.lower()))


def misquotes(text, src):
    """Quotes the model attributes ('the reference says: "..."') whose pieces (split at "...") are not in src (the row
    prompt, the question, the machine's outputs): Lightning invents prompt text, and a row must not teach that."""
    src = words(src)
    return [q for q in QUOTE.findall(text) if any(len(w.split()) >= 3 and w not in src
                                                    for w in map(words, re.split(r"\.\.\.|\u2026", q)))]


QUOTE = re.compile(r'\b(?:says?|said|states?|instructs?|tells? us|mentions?)\s*:?\s*"([^"\n]{15,300})"', re.I)


def clean(r, card=frozenset(), prompt=""):
    """card: grams(generating prompt) - grams(row prompt); prompt: the row prompt (quotes must come from it, the
    question or the machine's output blocks)."""
    think, _, answer = r["text"].rpartition("</think>")
    progs = [re.findall(F + r"forth\n(.*?)\n" + F, seg, re.S) for seg in (think, answer)]
    return (r["text"].count("</think>") == 1 and not any("..." in b["program"] for b in r["blocks"])
            and not META.search(r["text"]) and not any(VAR_IDIOM.search(b["program"]) for b in r["blocks"])
            and not misquotes(r["text"], prompt + " " + r["q"] + " " + " ".join(OUT_SPAN.findall(r["text"])))
            and not grams(OUT_SPAN.sub("", r["text"])) & (card - grams(r["q"]))
            and all(len(ps) == len({p.strip() for p in ps}) for ps in progs))


VAR_IDIOM = re.compile(r"(?<![\w-])-?\d+\s+variable\b", re.I)  # "19573 variable x !": a value before `variable` (copied idiom)
META = re.compile(r"\bas (per (the )?instructions?|instructed|required|told)\b|\b(user|prompt|instructions?|rules?|guidelines?|problem)\b"
                  r"[^.\n]{0,30}\b(says?|said|wants?|asks?|tells?|requires?|instructs?|specifies)\b[^.\n]{0,40}"
                  r"\buse (the )?(forth )?(machine|tool)", re.I)


def accepted_direct(r):
    return bool(r["ok"]) and "</think>" in r["text"] and not r["blocks"] and F + "forth" not in r["text"]


def prompt_text(rubric, q, tok):
    msgs = [{"role": "system", "content": rubric["system"]}]
    for uq, ua in rubric.get("shots", []):
        msgs += [{"role": "user", "content": uq}, {"role": "assistant", "content": ua}]
    msgs.append({"role": "user", "content": q})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def row(prompt, completion, tok, masked=()):
    """Composite-exact row: the token stream the in-weights kernel produces. The prompt and each model segment are
    tokenized on their own (the model's closing fence as '```\\n'), text_vm.stream inserts the kernel's visible output
    blocks after each fence (it carries state through the turn), the text after a block is tokenized fresh. Mask 1 on
    the model's tokens, 0 on the prompt, the kernel's tokens and blocks whose output is an error. Raises ValueError
    when the stream's text is not the rollout's (an output the current spec prints differently, a block the
    kernel would run that the rollout did not): the model's later text answered something else. masked: extra
    (start, end) character spans of the completion to mask (a demo's wrong first program that ran without an error)."""
    marks = list(OUT_SPAN.finditer(completion))  # loop_sim: fence + '\\n' + output block + '\\n' + continuation
    model, offs, k = [], [], 0
    for i in range(len(marks) + 1):
        last = i == len(marks)
        end = len(completion) if last else marks[i].start() - 1
        enc = tok(completion[k:end] + ("<|im_end|>" if last else "\n"), add_special_tokens=False,
                  return_offsets_mapping=True)
        model += enc.input_ids
        offs += [k + a for a, _ in enc.offset_mapping]
        k = len(completion) if last else marks[i].end()
    out, owned, _ = loop_sim.T.stream(model, tok)
    owned, errs, ids, mask, j = set(owned), [m.span() for m in ERR_SPAN.finditer(completion)] + list(masked), [], [], 0
    for p, t in enumerate(out):
        if p in owned:
            if t >= 1000:  # visible kernel text (hidden kernel tokens never reach the base)
                ids.append(t)
                mask.append(0)
            continue
        ids.append(t)
        mask.append(int(not any(s <= offs[j] < e for s, e in errs)))
        j += 1
    if tok.decode(ids) != completion + "<|im_end|>":
        raise ValueError("the kernel stream differs from the rollout")
    pids = tok(prompt, add_special_tokens=False).input_ids
    return pids + ids, [0] * len(pids) + mask


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--student", nargs="+", required=True)
    a.add_argument("--teacher", nargs="*", default=[])
    a.add_argument("--rubric", required=True)
    a.add_argument("--per-problem", type=int, default=2)
    a.add_argument("--all", action="store_true", help="every problem is a target, also the always-solved ones")
    a.add_argument("--direct", action="store_true", help="no-machine rows: accept correct answers without any block")
    a.add_argument("--either", action="store_true", help="accept machine-backed or direct answers (stage 2a)")
    a.add_argument("--per-family", type=int, default=10**9, help="cap rows per family")
    a.add_argument("--clean", metavar="CARD", help="drop transcripts quoting CARD (the generating prompt), reruns")
    a.add_argument("--teacher-too", action="store_true", help="teacher rows even where the student has its own")
    a.add_argument("--max-len", type=int, default=16384)
    a.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    a.add_argument("--out", required=True)
    args = a.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rubric = json.load(open(args.rubric)) if args.rubric.endswith(".json") else {"system": open(args.rubric).read()}
    acc = (lambda r: accepted(r) or accepted_direct(r)) if args.either else accepted_direct if args.direct else accepted
    if args.clean:
        gen = open(args.clean).read()
        gen = json.loads(gen)["system"] if args.clean.endswith(".json") else gen
        card = grams(gen) - grams(rubric["system"])
    ok = (lambda r: acc(r) and clean(r, card, rubric["system"])) if args.clean else acc
    stud, own = collections.defaultdict(list), collections.defaultdict(list)
    for f in args.student:
        for r in json.load(open(f)):
            stud[r["q"]].append(r["ok"])
            if ok(r):
                own[r["q"]].append(r | {"source": "student"})
    targets = {q for q, oks in stud.items() if args.all or not all(oks)}
    teach = collections.defaultdict(list)
    for f in args.teacher:
        for r in json.load(open(f)):
            if r["q"] in targets and ok(r):
                teach[r["q"]].append(r | {"source": "teacher"})
    good = {q: (own[q] + teach[q] if args.teacher_too else own[q] or teach[q]) for q in sorted(targets) if own[q] or teach[q]}  # sorted: the same rows every run
    rows, fam, src, too_long, differ = [], collections.Counter(), collections.Counter(), 0, 0
    ranked = {q: sorted(rs, key=lambda r: (-quality(r), len(r["text"])))[:args.per_problem] for q, rs in good.items()}
    for k in range(args.per_problem):  # every problem's best row before any problem's second (fair under --per-family)
        for q, rs in ranked.items():
            if k >= len(rs) or fam[rs[k].get("family")] >= args.per_family:
                continue
            r = rs[k]
            try:
                ids, mask = row(prompt_text(rubric, q, tok), r["text"], tok)
            except ValueError:
                differ += 1
                continue
            if len(ids) > args.max_len:
                too_long += 1
                continue
            rows.append({"input_ids": ids, "train_mask": mask, "q": q, "family": r.get("family"), "answer": r["a"],
                         "blocks": len(r["blocks"]), "retries": sum(b["output"].startswith("error") for b in r["blocks"]),
                         "source": r["source"], "quality": round(quality(r), 3)})
            fam[r.get("family")] += 1
            src[r["source"]] += 1
    with open(args.out, "w") as f:
        f.write("".join(json.dumps(x) + "\n" for x in rows))
    solved_all = len(stud) - len(targets)
    print(f"student: {len(stud)} problems, {solved_all} always solved (skipped), {len(targets)} targets")
    print(f"verified transcripts for {len(good)}/{len(targets)} targets ({sum(bool(own[q]) for q in targets)} with the "
          f"student's own); {len(rows)} rows by source {dict(src)} "
          f"({too_long} over {args.max_len} tokens, {differ} whose kernel outputs differ from the spec now dropped), {sum(r['retries'] > 0 for r in rows)} with retries")
    print(f"trained tokens {sum(map(sum, (r['train_mask'] for r in rows)))}, total tokens "
          f"{sum(len(r['input_ids']) for r in rows)}; by family {dict(fam)}")


_e = (F + "forth\n1 +\n" + F + "\n" + F + "output\nerror: stack underflow\n" + F + "\n"
      + F + "forth\n1 .\n" + F + "\n" + F + "output\n1\n" + F + "\n")
assert [m.group() for m in ERR_SPAN.finditer(_e)] == [F + "forth\n1 +\n" + F + "\n"]  # the fix after it stays trained
_r = {"ok": True, "text": "x</think>\nso \\boxed{4}", "blocks": [], "got": "4", "a": "4", "q": "2+2?"}
assert accepted_direct(_r) and not accepted_direct(_r | {"text": "x</think>\n" + F + "forth\n2 2 + .\n" + F})
_b = F + "forth\n1 .\n" + F + "\n"
assert clean(_r) and not clean(_r | {"text": _b + _b + "</think>"}) and not clean(_r | {"blocks": [{"program": "... code"}]})
assert clean(_r | {"text": _b + "</think>" + _b})  # tried in the thinking, then written in the answer: fine
_c = grams("Each block runs at once; its words and variables stay defined.") - grams("Use the machine.")
assert not clean(_r | {"text": "so: each block runs at once; its words and variables stay</think>"}, _c)
assert clean(_r | {"text": "x</think>" + F + "output\neach block runs at once; its words and variables stay\n" + F + "\n"}, _c)
assert not clean(_r | {"text": "a</think>b</think>"})
assert not clean(_r | {"text": "so I'll use the machine as instructed</think>"})
assert not clean(_r | {"text": "use a Forth machine (as per instructions)</think>"})
assert not clean(_r | {"text": "the user wants me to use the Forth machine</think>"})
assert clean(_r | {"text": "the prompt says: always end your answer with the final answer in boxed</think>"})
assert not clean(_r | {"blocks": [{"program": "19573 variable x !"}]}) and clean(_r | {"blocks": [{"program": "variable x 19573 x !"}]})
_p = "Use it only when the computation needs it. Numbers have up to 20 digits."
assert clean(_r | {"text": 'the prompt says "Use it only when the computation needs it" ... </think>'}, prompt=_p)
assert clean(_r | {"text": 'it says: "Use it only when ... up to 20 digits"</think>'}, prompt=_p)
assert not clean(_r | {"text": 'the reference says "do loops count i up to the limit"</think>'}, prompt=_p)
assert not accepted_direct(_r | {"ok": False}) and not accepted_direct(_r | {"text": "no close"}) and not accepted(_r)

if __name__ == "__main__":
    main()
