"""vLLM (0.29) out-of-tree model for the Lightning + Forth kernel composite (HF reference: modeling_forth_lightning.py).
Two stock vLLM NemotronHForCausalLM under prefixes `base.` and `kernel.`, so every Mamba / attention layer of each gets
its own cache slot (the cache shapes of both are identical: 64x64 Mamba heads, 8 groups, state 128, conv 4; 2 kv x 128).
  forward:        base(inputs_embeds with embed_override rows) and kernel(input_ids) -> hidden states concatenated
  kernel:         its Linears store only their nonzero block (Gathered), its inert attention is skipped (Zero)
  compute_logits: split; base logits with word_head columns (+ an lm_head LoRA's delta on them too); where the
                  kernel's max logit > mine: kernel logits x 2^16
                  (kernel logits from its nonzero lm_head rows only: the rest are exactly 0)
The base is read from config.base_model (its config.json is merged under the composite's, so vLLM sees Lightning's
sizes / quantization); only the base gets the quant config, the kernel stays bf16.
LoRA (SupportsLoRA): adapters in the base's PEFT / AutoModel names (base_model.model.model.layers.N.mixer.*) land on
base.*; kernel.* modules are never wrapped with LoRA (register() filters them) and kernel.* adapter tensors are skipped.
Visible stream (env FORTH_VISIBLE=1; HF semantics of modeling_forth_lightning.py): the base reads only the visible
stream. Hidden tokens (ids the kernel can emit below 1000, all <SPECIAL_n>) are masked out of the model's own rows, so
on any generated stream "id is hidden" == "the kernel emitted it, id < 1000"; vllm_visible.py builds the base's own
metadata (visible positions, no state / KV for hidden tokens) and the token order the base runs in; forward gathers the
base's inputs in that order and puts back, at every position, the base's state after the last visible token
(forth_visible op; across steps via b_last, one row per running request). The two backbones then compile separately
(own model tags); serve with prefix caching off, async scheduling off, cudagraph_mode FULL_DECODE_ONLY (asserted).
Exactness (kernel/composite/vllm_suite.sh, vllm_visible_check.py): chunk size = the kernel's (64), Gathered as bf16 GEMMs
with cuBLAS reduced-precision reduction off, prompts >= 2 tokens (a 1-token prompt runs as a cudagraph decode).
Registration: a `vllm.general_plugins` entry point from forth_vllm-0.dist-info next to this file, i.e. just
  PYTHONPATH=/srv/nemo/composite vllm serve /srv/nemo/forth-lightning ...   (or LLM(...) in Python)
"""

import contextlib
import copy
import glob
import os

import torch
from torch import nn
from transformers import NemotronHConfig, PretrainedConfig

OWNED = 2.0**16
VISIBLE = os.environ.get("FORTH_VISIBLE", "0") == "1"


class ForthLightningConfig(NemotronHConfig):
    """The base's NemotronH config (read from base_model) with the composite's keys on top."""
    model_type = "forth_lightning"

    @classmethod
    def from_dict(cls, config_dict, **kw):
        base, _ = PretrainedConfig.get_config_dict(config_dict["base_model"])
        # vLLM's Mamba2 metadata uses ONE chunk size (this config's) for every Mamba layer: the kernel's (64) must win,
        # its chunked scan is exact only up to it; the base's SSD math is the same at any chunk size
        chunk = min(base.get("chunk_size", 128), config_dict["kernel"].get("chunk_size", 128))
        return super().from_dict({**base, **config_dict, "chunk_size": chunk}, **kw)


def register():
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, NemotronHForCausalLMConfig
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    _CONFIG_REGISTRY["forth_lightning"] = ForthLightningConfig
    MODELS_CONFIG_MAP["ForthLightning"] = NemotronHForCausalLMConfig  # mamba_ssm_cache_dtype from config (float32)
    ModelRegistry.register_model("ForthLightning", "vllm_forth:ForthLightningForCausalLM")
    from vllm.lora.model_manager import LoRAModelManager
    # the kernel must stay exact: no LoRA layers inside it at all. shortcut: patches a private vLLM 0.29 method;
    # re-check on a vLLM upgrade
    match = LoRAModelManager._match_target_modules
    LoRAModelManager._match_target_modules = lambda self, name, m: not name.startswith("kernel.") and match(self, name, m)
    from vllm.v1.engine.input_processor import InputProcessor
    # vLLM 0.29 runs a 1-token first prefill as a FULL-cudagraph decode, which reads the request's uninitialised Mamba
    # state instead of zeros: the kernel breaks for the whole request. Real prompts (chat template) are longer.
    check = InputProcessor._validate_prompt_len

    def validate(self, n, prompt_type, *a, **k):
        if self.model_config.hf_config.model_type == "forth_lightning" and prompt_type == "decoder" and n < 2:
            raise ValueError("forth_lightning: prompts need >= 2 tokens (1-token prompts run as a cudagraph decode)")
        return check(self, n, prompt_type, *a, **k)
    InputProcessor._validate_prompt_len = validate
    if VISIBLE:
        import vllm_visible
        vllm_visible.patch_runner()
    if os.environ.get("FORTH_DEBUG"):  # the Mamba2 chunk size actually used, per build (prefill / decode)
        from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder as B
        build, seen = B.build, set()

        def logged(self, *a, **k):
            md = build(self, *a, **k)
            key = (self.chunk_size, md.num_prefills > 0, md.num_decodes > 0, tuple(self.layer_names[:1]))
            if key not in seen:
                seen.add(key)
                print("FORTH_DEBUG mamba2 build: chunk_size", md.chunk_size, "prefills", md.num_prefills, "decodes",
                      md.num_decodes, "layers", len(self.layer_names), self.layer_names[:2], flush=True)
            return md
        B.build = logged


def _forth_visible(hb: torch.Tensor, src: torch.Tensor, slot: torch.Tensor, upd_slot: torch.Tensor,
                   upd_src: torch.Tensor, b_last: torch.Tensor) -> torch.Tensor:
    """hb: the base's outputs in its own token order. Position j gets hb[src[j]] (its request's last visible token at
    or before j in this step), or b_last[slot[j]] (src -1: none yet in this step); then b_last[upd_slot] = hb[upd_src]."""
    n, m = hb.shape[0], b_last.shape[0] - 1
    out = torch.where((src >= 0)[:, None], hb[src.clamp(0, n - 1)], b_last[slot.clamp(0, m)])
    b_last.index_copy_(0, upd_slot.clamp(0, m), hb[upd_src.clamp(0, n - 1)])
    return out


def _forth_visible_fake(hb: torch.Tensor, src: torch.Tensor, slot: torch.Tensor, upd_slot: torch.Tensor,
                        upd_src: torch.Tensor, b_last: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(hb)


if VISIBLE:
    from vllm.utils.torch_utils import direct_register_custom_op
    # shortcut: b_last is mutated but not declared (a declared mutation gets functionalized and vLLM's graph split
    # then finds a cycle); safe while b_last is read and written only inside this op
    direct_register_custom_op("forth_visible", _forth_visible, mutates_args=[], fake_impl=_forth_visible_fake)


from vllm.compilation.backends import set_model_tag  # noqa: E402
from vllm.compilation.decorators import support_torch_compile  # noqa: E402
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator  # noqa: E402
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsLoRA  # noqa: E402
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM  # noqa: E402
from vllm.model_executor.models.utils import WeightsMapper  # noqa: E402
from vllm.model_executor.layers.linear import LinearBase  # noqa: E402
from vllm.model_executor.models.nemotron_h import NemotronHAttention  # noqa: E402


class Gathered(nn.Module):
    """A kernel Linear without its zeros (~99% of the kernel's weights): only the nonzero rows x cols block is stored;
    forward gathers the used input columns, multiplies, scatters into the used output rows. Exact: the dropped
    terms are x * 0, and the kernel's values are small dyadics (fp32 accumulation is order-independent), as long as
    cuBLAS never reduces split-K partials in bf16 (its default for these small odd shapes, batch-size dependent):
    the model turns allow_bf16_reduced_precision_reduction off (kernel/composite/gathered_check.py: then every Gathered
    output equals the full bf16 Linear's bit for bit). Not an fp32 product: inductor fuses the input's .float() into
    the norm before it and skips the bf16 rounding there."""

    def __init__(self, w):
        super().__init__()
        nz, self.out = w != 0, w.shape[0]
        rows, cols = nz.any(1).nonzero().flatten(), nz.any(0).nonzero().flatten()
        for k, v in (("rows", rows), ("cols", cols), ("w", w[rows][:, cols].contiguous())):
            self.register_buffer(k, v, persistent=False)

    def forward(self, x):
        y = x.new_zeros(*x.shape[:-1], self.out)
        y[..., self.rows] = nn.functional.linear(x[..., self.cols], self.w)
        return y, None


class Zero(nn.Module):  # the kernel's inert attention (o_proj all zero): its output is exactly 0
    def forward(self, hidden_states, **kw):
        return torch.zeros_like(hidden_states)


class ForthLightningForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsLoRA):
    get_mamba_state_shape_from_config = NemotronHForCausalLM.get_mamba_state_shape_from_config  # same for both
    get_mamba_state_dtype_from_config = NemotronHForCausalLM.get_mamba_state_dtype_from_config
    get_mamba_state_copy_func = NemotronHForCausalLM.get_mamba_state_copy_func
    # renames the base checkpoint's quantized / excluded layer names and LoRA adapter names onto base.* (load_weights
    # maps the weights itself)
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"backbone.": "base.model.", "model.": "base.model.",
                                                          "lm_head": "base.lm_head"})
    packed_modules_mapping = NemotronHForCausalLM.packed_modules_mapping
    embedding_modules = {}
    is_non_gated_moe = NemotronHForCausalLM.is_non_gated_moe
    lora_skip_prefixes = ["mtp.", "kernel.", "base.mtp."]

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        cfg = self.config = vllm_config.model_config.hf_config
        if not os.environ.get("FORTH_ALLOW_RPR"):  # the kernel's exactness (Gathered); FORTH_ALLOW_RPR=1: speed A/B only
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        tag = set_model_tag if VISIBLE else (lambda t: contextlib.nullcontext())
        with tag("forth_base"):  # visible mode: two compiled backbones, each its own compile / cudagraph pieces
            self.base = NemotronHForCausalLM(vllm_config=vllm_config, prefix=f"{prefix}base")
        kv = copy.copy(vllm_config)  # shares compilation_config (static forward context), own model / quant config
        kv.model_config = copy.copy(vllm_config.model_config)
        kv.model_config.hf_config = kv.model_config.hf_text_config = NemotronHConfig(**cfg.kernel)
        kv.quant_config = None
        with tag("forth_kernel"):
            self.kernel = NemotronHForCausalLM(vllm_config=kv, prefix=f"{prefix}kernel")
        # one graph for both backbones + overrides; visible mode: each backbone compiles on its own and the glue
        # (gather, forth_visible) runs eager / inside FULL cudagraphs (in one graph vLLM's split finds a cycle)
        if not VISIBLE:
            self.base.model.do_not_compile = self.kernel.model.do_not_compile = True
        self.embed_override = nn.Embedding(len(cfg.override_ids), cfg.hidden_size)
        self.word_head = nn.Linear(cfg.hidden_size, len(cfg.word_ids), bias=False)
        slot = torch.full((cfg.vocab_size,), -1, dtype=torch.long)
        slot[cfg.override_ids] = torch.arange(len(cfg.override_ids))
        self.register_buffer("slot", slot, persistent=False)
        self.register_buffer("words", torch.tensor(cfg.word_ids), persistent=False)
        if VISIBLE:  # filled by vllm_visible every step; idle values = every token visible, nothing kept
            from vllm.config import CUDAGraphMode
            # shortcut: piecewise cudagraphs of the two separately compiled backbones break the kernel in prefill
            # (eager, compiled without graphs and FULL decode graphs are exact); find why before enabling them
            assert vllm_config.compilation_config.cudagraph_mode in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY), \
                "FORTH_VISIBLE=1: run with compilation_config={'cudagraph_mode': 'FULL_DECODE_ONLY'}"
            sc = vllm_config.scheduler_config
            T, R = sc.max_num_batched_tokens + sc.max_num_seqs, sc.max_num_seqs
            for k, v in (("v_perm", torch.arange(T)), ("v_src", torch.arange(T)), ("v_slot", torch.full((T,), R)),
                         ("v_upd_slot", torch.full((R,), R)), ("v_upd_src", torch.zeros(R, dtype=torch.long))):
                self.register_buffer(k, v, persistent=False)
            self.register_buffer("b_last", torch.zeros(R + 1, cfg.hidden_size), persistent=False)  # row R: dummy

    def embed_input_ids(self, input_ids):
        s = self.slot[input_ids]
        return torch.where((s >= 0)[:, None], self.embed_override(s.clamp(min=0)), self.base.embed_input_ids(input_ids))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
        e = self.embed_input_ids(input_ids)
        hk = self.kernel.model(input_ids, positions)
        if not VISIBLE:
            return torch.cat([self.base.model(None, positions, None, e), hk], -1)
        n = input_ids.shape[0]
        hb = self.base.model(None, positions, None, e[self.v_perm[:n].clamp(max=n - 1)])  # NoPE: positions unused
        hb = torch.ops.vllm.forth_visible(hb, self.v_src[:n], self.v_slot[:n], self.v_upd_slot, self.v_upd_src,
                                          self.b_last)
        return torch.cat([hb, hk], -1)

    def compute_logits(self, h):
        hb, hk = h.split([self.config.hidden_size, h.shape[-1] - self.config.hidden_size], -1)
        hb = hb.contiguous()
        logits = self.base.compute_logits(hb)  # + the adapter's lm_head delta when an lm_head LoRA is loaded
        word = self.word_head(hb)
        lp = self.base.logits_processor
        if hasattr(lp, "lora_b_stacked"):  # the same delta on the word columns (vLLM's own per-request LoRA kernel)
            d = torch.zeros_like(logits)
            d = lp.punica_wrapper.add_lora_logits(d, hb, lp.lora_a_stacked, lp.lora_b_stacked, 1.0) or d
            word = word + d[:, self.words].to(word)
        logits = logits.float()
        logits[:, self.words] = word.float()
        if VISIBLE:  # the model never writes a hidden id itself (so hidden == kernel-emitted on generated streams)
            logits[:, self.hid] = -float("inf")
        k = torch.zeros_like(logits)
        k[:, self.krows] = nn.functional.linear(hk, self.khead).float()
        return torch.where(k.amax(-1, keepdim=True) > self.config.mine, k * OWNED, logits)

    def load_weights(self, weights):
        own, kern = dict(self.named_parameters()), []
        loaded = set()
        for n, w in weights:  # the composite's checkpoint: kernel.*, embed_override, word_head
            if n.startswith("kernel."):
                kern.append((n[len("kernel."):], w))
            else:
                own[n].data.copy_(w)
                loaded.add(n)
        loaded |= {f"kernel.{n}" for n in self.kernel.load_weights(kern)}
        files = sorted(glob.glob(os.path.join(self.config.base_model, "*.safetensors")))
        loaded |= {f"base.{n}" for n in self.base.load_weights(safetensors_weights_iterator(files, False))}
        for name, m in list(self.kernel.model.named_modules()):
            parent, _, attr = name.rpartition(".")
            if isinstance(m, NemotronHAttention):
                assert not m.o_proj.weight.any()
                setattr(self.kernel.model.get_submodule(parent), attr, Zero())
            elif isinstance(m, LinearBase) and attr in ("in_proj", "out_proj", "up_proj", "down_proj"):
                assert getattr(m, "bias", None) is None
                setattr(self.kernel.model.get_submodule(parent), attr, Gathered(m.weight))
        w = self.kernel.lm_head.weight  # the kernel emits ~270 tokens; its other lm_head rows are 0 (1 GB per step)
        self.krows = (w != 0).any(1).nonzero().flatten()
        self.khead = w[self.krows].clone()
        self.hid = self.krows[self.krows < 1000]  # the kernel's hidden ids (HF: HIDDEN_BELOW)
        assert int(self.hid.min()) >= 18, "a kernel-emitted id is not a <SPECIAL_n> token"
        self.kernel.lm_head = None  # 1 GB more for the caches
        return loaded


if not VISIBLE:
    ForthLightningForCausalLM = support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1})(
        ForthLightningForCausalLM)
