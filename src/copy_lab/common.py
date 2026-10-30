"""Shared model-loading and evaluation helpers for the copy_lab experiments."""

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def load_model(name="attn-only-2l", device="cpu", min_ctx=0):
    """Load a HookedTransformer in eval mode; return (net, pool, bos)."""
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    from transformer_lens import HookedTransformer

    net = HookedTransformer.from_pretrained(name, device=device)
    net.eval()
    if min_ctx > net.cfg.n_ctx:
        raise ValueError("Sequence exceeds the model's context window")
    if net.tokenizer is None:
        raise ValueError("This experiment requires a model with a tokenizer")
    bos = net.tokenizer.bos_token_id
    if bos is None:
        raise ValueError("This experiment requires a BOS token")
    special = set(net.tokenizer.all_special_ids) | {bos}
    pool = torch.tensor([i for i in range(net.cfg.d_vocab) if i not in special])
    return net, pool, bos


def per_token_losses(net, tokens, device="cpu", fwd_hooks=()):
    """Per-token loss one sequence at a time; returns (samples, n_pos - 1) on CPU."""
    kwargs = dict(return_type="loss", loss_per_token=True)
    fwd_hooks = list(fwd_hooks)
    with torch.inference_mode():
        if fwd_hooks:
            return torch.cat(
                [
                    net.run_with_hooks(row[None].to(device), fwd_hooks=fwd_hooks, **kwargs).cpu()
                    for row in tokens
                ]
            )
        return torch.cat([net(row[None].to(device), **kwargs).cpu() for row in tokens])
