"""AutoModel ChatDataset with loss only on model-written assistant tokens: the program's model-written tokens, the answer
sentence after it and Lightning's own <|im_end|>. Masked: kernel-owned positions (deterministic kernel output), the
template's assistant header (<|im_start|>assistant\\n<think></think>: the generation prompt at inference) and the
template's newline after <|im_end|>. The kernel's end-of-call token (sp_vm.DONE = 693) is not <|im_end|>, so the turn
ends only at the first <|im_end|> after the program.
Rows carry spans (one per assistant turn: ids = program + kernel tokens, owned = kernel offsets into it, lead = the
thinking text right before it, trained too) from training/render_sp.py; older rows carry one kernel_ids / kernel_owned.
No-tool rows carry plain_from (token index where the assistant's own text starts) and no spans.
Config: dataset._target_: masked_chat.KernelMaskedChatDataset, with PYTHONPATH=/srv/nemo/training.
"""

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset

IM_END = 11  # <|im_end|> (Lightning tokenizer)


class KernelMaskedChatDataset(ChatDataset):
    def __getitem__(self, idx):
        s, row = super().__getitem__(idx), self.dataset[idx]
        lab = s["labels"]  # labels are shifted: lab[i] is the target token at i
        keep, pos = [False] * len(lab), 0
        if row.get("plain_from") is not None:  # a no-tool row: loss on the whole final assistant turn
            start = row["plain_from"] - 1  # lab[i] is the target at i: the first answer token sits at plain_from - 1
            end = lab.index(IM_END, start)
            s["labels"] = [t if start <= i <= end else -100 for i, t in enumerate(lab)]
            return s
        spans = row.get("spans") or [{"ids": row["kernel_ids"], "owned": row["kernel_owned"], "lead": 0}]
        for sp in spans:  # one per assistant turn, in order
            ids = sp["ids"]
            start = next(a for a in range(pos, len(lab)) if lab[a:a + len(ids)] == ids)  # StopIteration = drift
            end = lab.index(IM_END, start + len(ids))  # ValueError = no <|im_end|> after the program
            keep[start - sp["lead"]:end + 1] = [True] * (end + 1 - start + sp["lead"])
            for o in sp["owned"]:
                keep[start + o] = False
            pos = end + 1
        s["labels"] = [t if k else -100 for t, k in zip(lab, keep)]
        return s


class TextLoopDataset(ChatDataset):
    """Text-mode kernel-loop rows from training/distill_rows.py, already tokenized: input_ids (Lightning's chat template:
    prompt, then the teacher's whole turn - reasoning, ```forth blocks, the kernel's ```output blocks, answer,
    <|im_end|>) and train_mask (1 = a target token: what the model writes; 0 = the prompt and the kernel's output blocks).
    Config: dataset._target_: masked_chat.TextLoopDataset, with PYTHONPATH=/srv/nemo/training."""

    def __getitem__(self, idx):
        from nemo_automodel.components.datasets.llm.formatting_utils import _package_tokenized_example
        row = self.dataset[idx]
        return _package_tokenized_example(self.tokenizer, list(row["input_ids"]), list(row["train_mask"]),
                                          getattr(self.tokenizer, "eos_token_id", 0), self.pad_token_id,
                                          self.seq_length, truncation=self.truncation, padding=self.padding)
