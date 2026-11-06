"""Can a NemotronH Mamba-2 mixer hold an exactly-overwritable 64-bit row? (feasibility probe)

Hidden features (bf16): h[0:64] new bits, h[64:128] old bits, h[128] write flag, h[129] = 1.
Head 0 = data row (state[:, n=0]), head 1 = norm ballast via D*x. Both in group 0.
Scheme "reset": A=-e^3, dt=32 on write (decay exp(-643)=0, overwrite), dt=0 on NOOP (decay 1).
Scheme "diff":  A=-exp(-200)=-0 (decay always 1), write x=32*(new-old).
Scheme "signed": A=-0, per-row (state index n) difference write with x = bf16(silu(u)) = +-1/4 from
  set/clear features (h[0:64] set, h[64:128] clear), B one-hot row (h[130+n]); rows 3 and 5, read row 5.
Readout: out[0:64] should equal the current bits exactly on every token.
"""

import json, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoConfig, DynamicCache
from transformers.models.nemotron_h import modeling_nemotron_h as M

torch.manual_seed(0)
dev = "cuda"
cfg = AutoConfig.from_pretrained("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
H, P, N, G = cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state_size, cfg.n_groups
I = H * P
NOOPS = 1000
CELL = {"reset": 2.0**15, "diff": 2.0**15, "signed": 2.0**8}
_g = torch.arange(1, 2**15, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()  # all +bf16
_g = _g[(_g > 0.05) & (_g < 4)]
_s = lambda u: F.silu(u.to(torch.bfloat16))  # silu on a bf16 tensor, bf16 result (as conv output)
ALPHA = _g[_s(_g) == 0.25][0].item()  # bf16 u with bf16(silu(u)) == +1/4 exactly
BETA = _g[_s(-_g) == -0.25][0].item()  # bf16 u with bf16(silu(-u)) == -1/4 exactly


def build(scheme):
    m = M.NemotronHMamba2Mixer(cfg, layer_idx=0).to(dev, torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        W = m.in_proj.weight  # rows: [z I | x I | B G*N | C G*N | dt H]
        xo, bo, co, do = I, 2 * I, 2 * I + G * N, 2 * I + 2 * G * N
        for p in range(P):
            W[p, 129] = 32  # z gate head 0 -> silu(32) = 32
            if scheme == "reset":
                W[xo + p, p] = 32
            elif scheme == "signed":
                W[xo + p, p], W[xo + p, 64 + p] = ALPHA, -BETA
            else:
                W[xo + p, p], W[xo + p, 64 + p] = 32, -32
        for p in range(32):  # ballast: 32 channels of head 1
            W[P + p, 129] = 32
            W[xo + P + p, 129] = 32
        if scheme == "signed":
            W[bo + 3, 133], W[bo + 5, 135] = 32, 32  # B[group0, row] = 32 * write-to-row
            W[co + 5, 129] = 32  # read row 5
        else:
            W[bo + 0, 128] = 32  # B[group0, n0] = 32 * write
            W[co + 0, 129] = 32  # C[group0, n0] = 32
        W[do + 0, 128], W[do:do + H, 129] = 232, -200  # dt head0: 32 on write, softplus(-200)=0
        m.conv1d.weight[:, 0, -1] = 1  # identity tap at current position
        m.A_log.fill_(3.0 if scheme == "reset" else -200.0)
        m.D[1] = 2.0**43  # y_ballast = 2^43*32 = 2^48, gated 2^53
        m.norm.weight.fill_(1)
        m.norm.weight[:P] = 2.0**51 / (CELL[scheme] * 32 * 32)  # cell*C*gate * rsqrt(32*2^106/512)=2^-51 -> 1
        m.out_proj.weight[:P, :P] = torch.eye(P)
    m.A_log.data, m.D.data, m.dt_bias.data = m.A_log.data.float(), m.D.data.float(), m.dt_bias.data.float()
    return m


def tokens(scheme):
    v1 = torch.randint(0, 2, (64,))
    v2 = v1.clone()
    v2[:32] = 1 - v2[:32]  # 32 flips, both directions
    seq, want = [], []
    old = torch.zeros(64, dtype=torch.long)
    other = torch.randint(0, 2, (64,))
    z = torch.zeros(64, dtype=torch.long)
    for v, n in ((v1, NOOPS), (v2, NOOPS)):
        if scheme == "signed":  # set/clear features, as an earlier circuit MLP would compute them
            seq.append((v * (1 - old), old * (1 - v), 5)); want.append(v)
            if n == NOOPS and old.sum() == 0:
                seq.append((other, z, 3)); want.append(v)  # neighbour row 3 must survive
        else:
            seq.append((v, old, 1)); want.append(v)
        seq += [(z, z, 0)] * n
        want += [v] * n
        old = v
    h = torch.zeros(len(seq), cfg.hidden_size)
    for t, (new, o, w) in enumerate(seq):
        h[t, :64], h[t, 64:128], h[t, 128], h[t, 129] = new, o, int(w > 0), 1
        if scheme == "signed" and w:
            h[t, 130 + w] = 1
    if scheme == "signed":
        return h.to(dev, torch.bfloat16)[None], torch.stack(want).float().to(dev), (v1, v2, other)
    return h.to(dev, torch.bfloat16)[None], torch.stack(want).float().to(dev), (v1, v2, None)


def mamba_inputs(m, h):
    """vLLM-style: in_proj, then (torch) identity conv + silu, split."""
    zxbcdt = m.in_proj(h)[0]
    z, xbc, dt = zxbcdt.split([I, I + 2 * G * N, H], -1)
    xbc = F.silu(xbc.float()).to(torch.bfloat16)  # identity conv tap; checked exact below
    x, B, C = xbc.split([I, G * N, G * N], -1)
    return z, x.view(-1, H, P), B.view(-1, G, N), C.view(-1, G, N), dt


def head(m, y, z):  # vLLM gated norm + out_proj
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
    o = rms_norm_gated(y.reshape(-1, I), m.norm.weight, None, z=z, eps=m.norm.variance_epsilon,
                       group_size=I // G, norm_before_gate=False)
    return m.out_proj(o)


def vllm_prefill(m, h):
    from vllm.model_executor.layers.mamba.ops.ssd_combined import mamba_chunk_scan_combined_varlen
    z, x, B, C, dt = mamba_inputs(m, h)
    T, cs = x.shape[0], cfg.chunk_size
    nch = (T + cs - 1) // cs
    out = torch.empty_like(x)
    A = -torch.exp(m.A_log)
    st = mamba_chunk_scan_combined_varlen(
        x, dt, A, B, C, cs, torch.tensor([0, T], device=dev, dtype=torch.int32),
        torch.tensor([min(i * cs, T) for i in range(nch + 1)], device=dev, dtype=torch.int32),
        torch.tensor([nch - 1], device=dev, dtype=torch.int32),
        torch.zeros(nch, device=dev, dtype=torch.int32), out, D=m.D, dt_bias=m.dt_bias,
        dt_softplus=True, dt_limit=(0.0, float("inf")), state_dtype=torch.float32)
    return head(m, out, z), st[-1]


def vllm_step(m, h):
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
    z, x, B, C, dt = mamba_inputs(m, h)
    state = torch.zeros(1, H, P, N, device=dev, dtype=torch.float32)
    A = (-torch.exp(m.A_log))[:, None, None].expand(H, P, N)
    D, dtb = m.D[:, None].expand(H, P), m.dt_bias[:, None].expand(H, P)
    ys = []
    for t in range(x.shape[0]):
        y = torch.empty_like(x[t:t + 1])
        selective_state_update(state, x[t:t + 1], dt[t:t + 1, :, None].expand(1, H, P), A, B[t:t + 1],
                               C[t:t + 1], D, dtb, dt_softplus=True, out=y)
        ys.append(y)
    return head(m, torch.cat(ys), z), state[0]


def hf_prefill(m, h, seg=256):
    # shortcut: torch fallback materialises (chunks,128,128,heads,N) fp32, so feed 256-token pieces via the cache
    c = DynamicCache(config=cfg)
    outs = [m(h[:, i:i + seg], cache_params=c)[0] for i in range(0, h.shape[1], seg)]
    return torch.cat(outs), c.layers[0].recurrent_states[0][0]


def hf_step(m, h):
    c = DynamicCache(config=cfg)
    outs = [m(h[:, t:t + 1], cache_params=c)[0] for t in range(h.shape[1])]
    return torch.cat(outs), c.layers[0].recurrent_states[0][0]


def check(name, fn, m, h, want, expect_state):
    t0 = time.time()
    try:
        with torch.no_grad():
            out, st = fn(m, h)
    except Exception as e:  # report, keep going
        return {"error": f"{type(e).__name__}: {e}"[:400]}
    got = out[:, :64].float()
    err = (got - want).abs()
    bad = (err > 0).any(1).nonzero()
    r = {
        "readout_max_abs_err": err.max().item(),
        "tokens_wrong": int((err > 0).any(1).sum()),
        "first_wrong_token": int(bad[0]) if len(bad) else None,
        "head0_state_max_abs_err": (st[0].float() - expect_state[0]).abs().max().item(),
        "other_heads_max_abs": (st.float() - expect_state)[1:].abs().max().item(),
        "state_dtype": str(st.dtype),
        "seconds": round(time.time() - t0, 2),
    }
    print(name, r, flush=True)
    return r


if __name__ == "__main__":  # importable: prim_mixer.py reuses the path functions
    res = {"config": {k: getattr(cfg, k) for k in ("mamba_num_heads", "mamba_head_dim", "ssm_state_size",
                                                   "n_groups", "conv_kernel", "chunk_size", "time_step_min",
                                                   "mamba_ssm_cache_dtype")},
           "hf_time_step_limit": None, "noops_after_each_write": NOOPS}
    res["signed_x"] = {"alpha": ALPHA, "beta": BETA, "silu_bf16(alpha)": _s(torch.tensor(ALPHA)).item(),
                       "silu_bf16(-beta)": _s(torch.tensor(-BETA)).item()}
    h, want, (v1, v2, _) = tokens("reset")
    res["flips"] = {"0->1": int(((v1 == 0) & (v2 == 1)).sum()), "1->0": int(((v1 == 1) & (v2 == 0)).sum())}
    res["silu_exact"] = {str(v): F.silu(torch.tensor(float(v))).item() for v in (32, 0, -32)}
    res["exp_underflow"] = {"exp(-200)": torch.exp(torch.tensor(-200.0)).item(),
                            "softplus(-200)": F.softplus(torch.tensor(-200.0)).item(),
                            "exp(32*-e^3)": torch.exp(torch.tensor(32 * -torch.e**3)).item()}
    for scheme in ("reset", "diff", "signed"):
        h, want, (v1, v2, other) = tokens(scheme)
        m = build(scheme)
        res["hf_time_step_limit"] = list(m.time_step_limit)
        final = torch.zeros(H, P, N, device=dev)
        if scheme == "signed":
            final[0, :, 5], final[0, :, 3] = CELL[scheme] * v2.float(), CELL[scheme] * other.float()
        else:
            final[0, :, 0] = CELL[scheme] * v2.float()
        res[scheme] = {name: check(f"{scheme}/{name}", fn, m, h, want, final)
                       for name, fn in (("vllm_prefill_chunked", vllm_prefill), ("vllm_recurrent_step", vllm_step),
                                        ("hf_torch_prefill_chunked", hf_prefill), ("hf_torch_recurrent_step", hf_step))}
        m.time_step_limit = (0.0, float("inf"))  # what vLLM uses; HF hardcodes (time_step_min, inf)
        res[scheme]["hf_torch_prefill_chunked_dtlimit0"] = check(f"{scheme}/hf_prefill_dtlimit0", hf_prefill,
                                                                 m, h, want, final)
        del m
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
