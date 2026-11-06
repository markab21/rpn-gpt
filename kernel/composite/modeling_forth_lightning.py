"""Lightning + Forth kernel composite (trust_remote_code): an untouched base LM (stock NemotronH Lightning, loaded from
its own checkpoint by path / HF id) next to the compiled stored-program kernel (kernel/sp_kernel.py) as a narrow
submodule. Both read the same token stream, each with its own cache.

Ownership: the kernel's logits are exactly 0 except where it emits (its token's logit >= 1), so it owns the next token
iff its max logit > config.mine (0.5). There the composite's logits are the kernel's x OWNED (2^16): deterministic under
any sampler. Elsewhere they are the base's logits, bit for bit, except the word columns (below).
Base-side rows (the composite's own small weights; the base checkpoint stays byte-identical):
  embed_override: embedding rows for the kernel's reserved tokens (words the model writes, kernel-emitted tokens),
                  fed to the base as inputs_embeds;
  word_head:      lm_head rows for the words the model writes (<lit>, <call>, :, ;, IF, ...).
A PEFT LoRA on base.lm_head (e2e_demo.py attach_lora) also adds its delta to the word columns (_head_lora_delta).
Both are plain modules (PEFT modules_to_save / AutoModel can train them); the kernel must stay frozen.
Visible stream: the base reads only what its training data shows (model text + the visible output block). A token the
kernel emitted (its logit for that token at the previous position > mine) with an id < HIDDEN_BELOW is a hidden protocol
token (a special token: decodes to '' with skip_special_tokens) and goes to the kernel only; the base's cache does not
advance on it, and its logits there are the base's at the last token it read. Every other token goes to both. The base
runs one row at a time (rows skip different positions; Mamba state cannot skip a masked token exactly), pads
(attention_mask 0) skipped too, so each row's base sees exactly its unpadded visible tokens.
  m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
"""

import torch
from torch import nn
from transformers import (AutoModelForCausalLM, GenerationMixin, NemotronHConfig, NemotronHForCausalLM,
                          PretrainedConfig, PreTrainedModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

OWNED = 2.0**16  # kernel-owned logits are the kernel's (0 or >= 1) times this; base logits never come near it
HIDDEN_BELOW = 1000  # kernel-emitted ids below this are hidden protocol tokens; its output block ids are all >= 1000


class ForthLightningConfig(PretrainedConfig):
    model_type = "forth_lightning"

    def __init__(self, base_model=None, kernel=None, override_ids=(), word_ids=(), base_hidden_size=2688,
                 vocab_size=131072, mine=0.5, **kw):
        self.base_model, self.kernel, self.mine = base_model, kernel, mine
        self.override_ids, self.word_ids = list(override_ids), list(word_ids)
        self.base_hidden_size, self.vocab_size = base_hidden_size, vocab_size
        kw.setdefault("tie_word_embeddings", False)
        super().__init__(**kw)


class CompositeCache:
    """The kernel's cache, the base's per row (each a NemotronH DynamicCache: Mamba + attention layers; the base's
    covers only that row's visible tokens), and both models' last hidden states (B, 1, H): the kernel's decides whether
    the next token is one it emitted, the base's gives its logits while it skips."""
    is_compileable = False

    def __init__(self, base, kernel, k_last, b_last):
        self.base, self.kernel, self.k_last, self.b_last = base, kernel, k_last, b_last

    def get_seq_length(self, layer_idx=0):
        return self.kernel.get_seq_length()  # the full stream (the base's caches are shorter)


class ForthLightning(PreTrainedModel, GenerationMixin):
    config_class = ForthLightningConfig
    _is_stateful = True
    _tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.kernel = NemotronHForCausalLM(NemotronHConfig(**config.kernel))
        self.embed_override = nn.Embedding(len(config.override_ids), config.base_hidden_size)
        self.word_head = nn.Linear(config.base_hidden_size, len(config.word_ids), bias=False)
        self.base = None  # attached by from_pretrained (never saved with the composite)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False  # forward builds its CompositeCache

    @classmethod
    def from_pretrained(cls, path, *args, base_model=None, base_kwargs=None, **kw):
        m = super().from_pretrained(path, *args, **kw)
        bk = {k: kw[k] for k in ("dtype", "torch_dtype", "device_map", "attn_implementation") if k in kw}
        m.base = AutoModelForCausalLM.from_pretrained(base_model or m.config.base_model, **bk | (base_kwargs or {}))
        # HF's NemotronH prefill clamps dt below at time_step_min (its decode, vLLM and AutoModel don't): 0.157 -> 0.030
        # |dlogp|/token vs vLLM BF16 without it. Base only: the kernel is built for (and verified with) the clamp.
        for mx in m.base.modules():
            if hasattr(mx, "time_step_limit"):
                mx.time_step_limit = (0.0, float("inf"))
        return m

    def save_pretrained(self, *args, **kw):
        base = self._modules.pop("base", None)
        try:
            super().save_pretrained(*args, **kw)
        finally:
            self._modules["base"] = base

    def get_input_embeddings(self):
        return self.base.get_input_embeddings() if self.base is not None else None

    def _ids(self, device):
        if getattr(self, "_slot", None) is None or self._slot.device != device:
            self._slot = torch.full((self.config.vocab_size,), -1, dtype=torch.long, device=device)
            self._slot[self.config.override_ids] = torch.arange(len(self.config.override_ids), device=device)
            self._words = torch.tensor(self.config.word_ids, device=device)
        return self._slot, self._words

    def _head_lora_delta(self, h, words):
        """Word columns of a PEFT LoRA on the base's lm_head (0 without one): the adapter was trained on a base copy
        whose lm_head rows at word_ids ARE word_head (training/forthrows.py), so its low-rank delta applies to every
        column, word columns included (merged or not: word_head replaces the word columns, so the delta is re-added)."""
        lm = self.base.lm_head
        if not hasattr(lm, "lora_A") or getattr(lm, "disable_adapters", False):
            return 0.0
        d = 0.0
        for a in lm.active_adapters:
            if a in lm.lora_A:
                A, B = lm.lora_A[a].weight, lm.lora_B[a].weight
                d = d + lm.scaling[a] * torch.nn.functional.linear(h.to(A.dtype) @ A.T, B[words]).float()
        return d

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None,
                logits_to_keep=0, **kw):
        pc = past_key_values
        kin = input_ids.to(self.kernel.device)
        ko = self.kernel.model(input_ids=kin, attention_mask=attention_mask, past_key_values=pc and pc.kernel,
                               use_cache=use_cache)
        hk = ko.last_hidden_state
        # hidden: the kernel emitted this token (its logit for it one position earlier > mine) and it is a protocol id
        prev = torch.cat([pc.k_last if pc is not None else torch.zeros_like(hk[:, :1]), hk[:, :-1]], 1)
        emitted = (prev.float() * self.kernel.lm_head.weight[kin].float()).sum(-1) > self.config.mine
        keep = ~(emitted & (kin < HIDDEN_BELOW)).to(input_ids.device)
        if attention_mask is not None:
            keep &= attention_mask[:, -input_ids.shape[1]:].bool()
        slot, words = self._ids(input_ids.device)
        emb = self.base.get_input_embeddings()(input_ids)
        s = slot[input_ids]
        emb = torch.where((s >= 0)[..., None], self.embed_override(s.clamp(min=0)).to(emb.dtype), emb)
        hb, caches = [], []
        for b in range(input_ids.shape[0]):  # one row at a time: each skips its own positions
            c = pc.base[b] if pc is not None else None
            h = pc.b_last[b] if pc is not None else emb.new_zeros(1, emb.shape[-1])
            if keep[b].any():
                bo = self.base.model(inputs_embeds=emb[b:b + 1, keep[b]], past_key_values=c, use_cache=use_cache)
                c, h = bo.past_key_values, torch.cat([h, bo.last_hidden_state[0]])
            hb.append(h[keep[b].cumsum(0)])  # each position: the base's state after the last token it read
            caches.append(c)
        hb = torch.stack(hb)
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        h = hb[:, sl]
        logits = self.base.lm_head(h).float()
        logits[..., words] = (self.word_head(h.to(self.word_head.weight.dtype)).float().to(logits.device)
                              + self._head_lora_delta(h, words))
        k = self.kernel.lm_head(hk[:, sl]).float().to(logits.device)
        logits = torch.where(k.amax(-1, keepdim=True) > self.config.mine, k * OWNED, logits)
        cache = (CompositeCache(caches, ko.past_key_values, hk[:, -1:], hb[:, -1:])
                 if ko.past_key_values is not None else None)
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)


ForthLightningConfig.register_for_auto_class()
ForthLightning.register_for_auto_class("AutoModelForCausalLM")
