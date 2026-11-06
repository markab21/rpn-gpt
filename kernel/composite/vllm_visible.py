"""The base's visible-only view inside vLLM 0.29's GPUModelRunner (FORTH_VISIBLE=1; model side: vllm_forth.py).
Each step _build_attention_metadata runs twice: once as usual (the kernel's Mamba layers keep that metadata), once on a
base view swapped into the runner's state (all attention layers + the base's Mamba layers use it; the kernel's
attention is Zero and reads nothing). Per request r with q scheduled tokens, vc visible tokens already read:
  real   (>= 1 visible scheduled token): its visible tokens, at visible positions vc.., in the base's token order;
  inert  (none visible): one token kept so the batch keeps its shape; Mamba state -> the null block, no KV write;
         a decode stays a decode (seq len vc), a prefill row becomes a fresh 1-token prefill.
Base rows are ordered decode-like (1 token, prior state) < fresh 1-token < longer, as vLLM's builders require; hidden
tokens not kept go to the tail (computed, never read). Hidden = token id in the kernel's emittable set below 1000 (the
model is masked off those ids). Needs: no async scheduling, no prefix caching, no spec decode / DBO / DCP, one LoRA
per batch (the base's token order is not the scheduler's; one adapter makes the LoRA token mapping order-free).
"""

import numpy as np
import torch

NULL_BLOCK = 0  # vLLM's reserved null block (NULL_BLOCK_ID): state reads / writes there are scratch


class _Buf:
    def __init__(self, cpu, gpu):
        self.cpu, self.gpu, self.np = cpu, gpu, cpu.numpy()


class _Tables:
    """Stands in for input_batch.block_table during the base-view build."""

    def __init__(self, tabs):
        self.tabs = tabs

    def __getitem__(self, g):
        return self.tabs[g]


class _Tab:
    def __init__(self, gpu):
        self.gpu = gpu

    def get_device_tensor(self, n):
        return self.gpu[:n]


class View:
    def __init__(self, runner, model):
        from vllm.v1.kv_cache_interface import MambaSpec
        r = runner
        assert not r.use_async_scheduling, "FORTH_VISIBLE needs async_scheduling=False (hidden ids are read on the host)"
        assert not r.cache_config.enable_prefix_caching, "FORTH_VISIBLE needs prefix caching off"
        assert r.speculative_config is None and r.dcp_world_size == 1
        self.r, self.m, dev = r, model, r.device
        self.lut = np.zeros(r.model_config.get_vocab_size(), bool)
        self.lut[model.hid.cpu().numpy()] = True
        n = r.query_start_loc.cpu.shape[0]
        mk = lambda dt: torch.zeros(n, dtype=dt).pin_memory()
        self.qsl = _Buf(mk(torch.int32), torch.zeros(n, dtype=torch.int32, device=dev))
        self.seq_lens = torch.zeros_like(r.seq_lens)
        self.opt = torch.zeros_like(r.optimistic_seq_lens_cpu)
        self.nc = torch.zeros_like(r.input_batch.num_computed_tokens_cpu_tensor)
        self.npr = torch.zeros_like(r.input_batch.num_prompt_tokens_cpu_tensor)
        groups = r.kv_cache_config.kv_cache_groups
        self.mamba = {g for g, kg in enumerate(groups) if isinstance(kg.kv_cache_spec, MambaSpec)}
        self.attn = [g for g in range(len(groups)) if g not in self.mamba]
        bt = r.input_batch.block_table
        self.tabs = _Tables([_Tab(torch.zeros_like(bt[g].block_table.gpu)) for g in range(len(groups))])
        self.kernel_mamba = {name for g in self.mamba for name in groups[g].layer_names if name.startswith("kernel.")}
        # the base's Mamba layers get their own builders: fresh ones, so no buffer (cudagraph or prefill staging) is
        # shared with the kernel's metadata of the same step
        self.builders = {}
        for g in self.mamba:
            for ag in r.attn_groups[g]:
                b = ag.get_metadata_builder(0)
                self.builders[id(ag)] = [type(b)(b.kv_cache_spec, b.layer_names, b.vllm_config, b.device)]
        self.R = model.b_last.shape[0] - 1
        self.rows, self.vc = {}, {}  # req_id -> b_last row; req_id -> (num computed tokens, visible among them)

    def visible_before(self, i, rid, nc):
        known = self.vc.get(rid)
        if known is None or known[0] != nc:  # new, preempted / resumed: count from the token ids
            known = (nc, int(np.count_nonzero(~self.lut[self.r.input_batch.token_ids_cpu[i, :nc]])))
        return known[1]

    def row(self, rid):
        if rid not in self.rows:
            free = sorted(set(range(self.R)) - set(self.rows.values()))
            self.rows[rid] = free[0]
        return self.rows[rid]

    def plan(self, num_reqs, num_tokens):
        """Host side of one step: base rows, token order, metadata values, model-side gather / keep arrays."""
        r, ib = self.r, self.r.input_batch
        live = set(r.requests)  # every request the runner still holds (scheduled this step or not)
        for rid in [k for k in self.rows if k not in live]:
            del self.rows[rid]
            self.vc.pop(rid, None)
        lora = getattr(ib, "request_lora_mapping", None)
        if lora is not None and num_reqs:
            assert len(set(lora[:num_reqs].tolist())) == 1, "FORTH_VISIBLE: one LoRA per batch (see module doc)"
        qs = r.query_start_loc.np[:num_reqs + 1].astype(np.int64)
        ids = r.input_ids.cpu[:num_tokens].numpy()
        hid = self.lut[ids]
        nc = ib.num_computed_tokens_cpu[:num_reqs]
        pre = nc < ib.num_prompt_tokens[:num_reqs]
        rows = []  # (sort key, orig row, tokens kept, visible before, seq len, computed, prefilling, inert)
        for i in range(num_reqs):
            rid = ib.req_ids[i]
            vb = self.visible_before(i, rid, int(nc[i]))
            tok = qs[i] + np.flatnonzero(~hid[qs[i]:qs[i + 1]])
            q = len(tok)
            self.vc[rid] = (int(nc[i]) + int(qs[i + 1] - qs[i]), vb + q)
            if q == 0 and not pre[i]:
                rows.append((0, i, [qs[i]], vb, max(vb, 1), max(vb, 1) - 1, False, True))
            elif q == 0:
                rows.append((1, i, [qs[i]], 0, 1, 0, True, True))
            else:
                key = 0 if q == 1 and vb > 0 else 1 if q == 1 else 2
                rows.append((key, i, tok, vb, vb + q, vb, bool(pre[i]), False))
        rows.sort(key=lambda x: x[0])  # stable
        perm = np.concatenate([np.asarray(x[2], np.int64) for x in rows]) if rows else np.zeros(0, np.int64)
        T = len(perm)
        rest = np.setdiff1d(np.arange(num_tokens), perm, assume_unique=True)
        src = np.full(num_tokens, -1, np.int64)
        slot = np.full(num_tokens, self.R, np.int64)
        upd_slot, upd_src = np.full(self.R, self.R, np.int64), np.zeros(self.R, np.int64)
        brow, bpos = np.zeros(T, np.int64), np.full(T, -1, np.int64)  # per base token: orig row, visible position
        k = 0
        for j, (_, i, tok, vb, *_, inert) in enumerate(rows):
            b = self.row(ib.req_ids[i])
            slot[qs[i]:qs[i + 1]] = b
            if not inert:
                at = np.full(qs[i + 1] - qs[i], -1, np.int64)  # latest visible token at or before each position
                at[np.asarray(tok) - qs[i]] = np.arange(k, k + len(tok))
                src[qs[i]:qs[i + 1]] = np.maximum.accumulate(at)
                upd_slot[j], upd_src[j] = b, k + len(tok) - 1
                bpos[k:k + len(tok)] = vb + np.arange(len(tok))
            brow[k:k + len(tok)] = i
            k += len(tok)
        return dict(rows=rows, perm=np.concatenate([perm, rest]), T=T, src=src, slot=slot, upd_slot=upd_slot,
                    upd_src=upd_src, brow=brow, bpos=bpos)

    def stage(self, p, num_reqs, num_tokens, num_reqs_padded, num_tokens_padded):
        """Writes the plan into the base-view buffers (swapped in for the second build) and the model's arrays."""
        r, m, dev = self.r, self.m, self.r.device
        rows = p["rows"]
        n = len(rows)
        q = np.array([len(x[2]) for x in rows], np.int32)
        self.qsl.np[0] = 0
        self.qsl.np[1:n + 1] = np.cumsum(q)
        self.qsl.np[n + 1:] = self.qsl.np[n] if n else 0
        self.qsl.gpu.copy_(self.qsl.cpu, non_blocking=True)
        sl = torch.zeros(self.opt.shape[0], dtype=self.opt.dtype)
        sl[:n] = torch.tensor([x[4] for x in rows], dtype=sl.dtype)
        self.opt.copy_(sl)
        self.seq_lens.copy_(sl.to(self.seq_lens.dtype), non_blocking=True)
        self.nc.zero_()
        self.npr.zero_()
        self.nc[:n] = torch.tensor([x[5] for x in rows], dtype=self.nc.dtype)
        self.npr[:n] = torch.tensor([x[5] + (len(x[2]) if x[6] else 0) for x in rows], dtype=self.npr.dtype)
        order = torch.tensor([x[1] for x in rows], dtype=torch.long).to(dev, non_blocking=True)
        inert = torch.tensor([x[7] for x in rows], dtype=torch.bool).to(dev, non_blocking=True)
        bt = r.input_batch.block_table
        for g, tab in enumerate(self.tabs.tabs):
            src = bt[g].block_table.gpu
            if n:
                tab.gpu[:n] = torch.where(inert[:, None], NULL_BLOCK, src[order])
        brow = torch.from_numpy(p["brow"]).to(dev, non_blocking=True)
        bpos = torch.from_numpy(p["bpos"]).to(dev, non_blocking=True)
        T = p["T"]
        for g in self.attn:  # base KV slots at visible positions, in place (only base attention writes KV)
            blk = bt[g]
            sm = blk.slot_mapping.gpu
            sm[:num_tokens_padded].fill_(-1)
            if T:
                bs = blk.block_size
                s = blk.block_table.gpu[brow, bpos.clamp(min=0) // bs].long() * bs + bpos.clamp(min=0) % bs
                sm[:T] = torch.where(bpos >= 0, s, -1)
        perm = np.concatenate([p["perm"], np.arange(num_tokens, num_tokens_padded)])
        for buf, a in ((m.v_perm, perm), (m.v_src, p["src"]), (m.v_slot, p["slot"]),
                       (m.v_upd_slot, p["upd_slot"]), (m.v_upd_src, p["upd_src"])):
            buf[:len(a)].copy_(torch.from_numpy(a), non_blocking=True)
        if num_tokens_padded > num_tokens:
            m.v_src[num_tokens:num_tokens_padded].copy_(m.v_perm[num_tokens:num_tokens_padded])
            m.v_slot[num_tokens:num_tokens_padded].fill_(self.R)
        return n, T, int(q.max()) if n else 1

    def idle(self, num_reqs_padded, num_tokens_padded):
        """Capture / dummy runs: the base view = the runner's own values, every token visible, nothing kept."""
        r, m, ib = self.r, self.m, self.r.input_batch
        self.qsl.cpu.copy_(r.query_start_loc.cpu)
        self.qsl.gpu.copy_(r.query_start_loc.gpu)
        self.seq_lens.copy_(r.seq_lens)
        self.opt.copy_(r.optimistic_seq_lens_cpu)
        self.nc.copy_(ib.num_computed_tokens_cpu_tensor)
        self.npr.copy_(ib.num_prompt_tokens_cpu_tensor)
        for g, tab in enumerate(self.tabs.tabs):
            tab.gpu.copy_(ib.block_table[g].block_table.gpu)
        m.v_perm.copy_(torch.arange(m.v_perm.shape[0], device=m.v_perm.device))
        m.v_src.copy_(m.v_perm)
        m.v_slot.fill_(self.R)
        m.v_upd_slot.fill_(self.R)

    def swapped(self):
        r, ib = self.r, self.r.input_batch
        saved = [(r, "query_start_loc", self.qsl), (r, "seq_lens", self.seq_lens), (r, "optimistic_seq_lens_cpu", self.opt),
                 (ib, "num_computed_tokens_cpu_tensor", self.nc), (ib, "num_prompt_tokens_cpu_tensor", self.npr),
                 (ib, "block_table", self.tabs)]
        saved += [(ag, "metadata_builders", self.builders[id(ag)]) for g in self.mamba for ag in r.attn_groups[g]]
        return _Swap(saved)


class _Swap:
    def __init__(self, items):
        self.items = items

    def __enter__(self):
        self.old = [(o, k, getattr(o, k)) for o, k, _ in self.items]
        for o, k, v in self.items:
            setattr(o, k, v)

    def __exit__(self, *a):
        for o, k, v in self.old:
            setattr(o, k, v)


def patch_runner():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    orig = GPUModelRunner._build_attention_metadata

    def build(self, num_tokens, num_reqs, max_query_len, num_tokens_padded=None, num_reqs_padded=None,
              ubatch_slices=None, for_cudagraph_capture=False, **kw):
        md, spec = orig(self, num_tokens, num_reqs, max_query_len, num_tokens_padded, num_reqs_padded, ubatch_slices,
                        for_cudagraph_capture=for_cudagraph_capture, **kw)
        m = self.get_model()
        if not hasattr(m, "b_last") or not md:
            return md, spec
        assert ubatch_slices is None, "FORTH_VISIBLE: no DBO"
        sig = (id(self.kv_cache_config), tuple(id(ag) for gs in self.attn_groups for ag in gs))
        v = getattr(self, "_forth_view", None)
        if v is None or v.sig != sig:  # (re)built once the KV caches / metadata builders exist (profiling runs first)
            v = self._forth_view = View(self, m)
            v.sig = sig
        ntp, nrp = num_tokens_padded or num_tokens, num_reqs_padded or num_reqs
        if for_cudagraph_capture or self.input_batch.num_reqs == 0:
            v.idle(nrp, ntp)
            T, mq = num_tokens, max_query_len
        else:
            nb, T, mq = v.stage(v.plan(num_reqs, num_tokens), num_reqs, num_tokens, nrp, ntp)
            assert nb == num_reqs and (num_tokens_padded is None or T == num_tokens), (nb, num_reqs, T, num_tokens)
        with v.swapped():  # T base tokens (hidden ones not kept sit after them, like padding)
            mb, _ = orig(self, T, num_reqs, mq, num_tokens_padded, num_reqs_padded, None,
                         for_cudagraph_capture=for_cudagraph_capture, **kw)
        return {k: (md[k] if k in v.kernel_mamba else mb[k]) for k in md}, spec

    GPUModelRunner._build_attention_metadata = build
