"""Score attention heads for induction/previous-token behavior and ablate the top ones."""

import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch

from copy_lab.common import ROOT, load_model, per_token_losses
from copy_lab.experiment import block_slices, make_inputs, summarize

PREVIOUS_OFFSET = -1


def induction_offset(length):
    """Offset from a scored query to the key holding the token it should copy."""
    return 1 - length


def offset_score(patterns, length, offset):
    """Mean attention from each scored second-block query q to key q + offset.

    patterns: (batch, heads, n_pos, n_pos), n_pos = 2 * length + 1. Returns (heads,).
    """
    if length < 2:
        raise ValueError("Need length >= 2")
    n_pos = 2 * length + 1
    if tuple(patterns.shape[-2:]) != (n_pos, n_pos):
        raise ValueError(
            f"Expected {n_pos}x{n_pos} attention patterns for length={length}, "
            f"got {tuple(patterns.shape[-2:])}"
        )
    second = block_slices(length)[1]
    q = torch.arange(second.start, second.stop)
    k = q + offset
    if bool((k < 0).any()) or bool((k >= n_pos).any()):
        raise ValueError("Offset key index out of range")
    return patterns[:, :, q, k].mean(dim=(0, 2))


def rank_heads(scores):
    """(layers, heads) scores -> [(layer, head), ...] sorted by score, descending."""
    scores = torch.as_tensor(scores)
    layers, heads = scores.shape
    pairs = [(layer, head) for layer in range(layers) for head in range(heads)]
    return sorted(pairs, key=lambda lh: scores[lh[0], lh[1]].item(), reverse=True)


def select_control_heads(ablated, n_heads, seed):
    """Pick random, disjoint control heads with the same per-layer counts as `ablated`."""
    by_layer = defaultdict(list)
    for layer, head in ablated:
        by_layer[layer].append(head)
    generator = torch.Generator().manual_seed(seed)
    control = []
    for layer in sorted(by_layer):
        ablated_heads = set(by_layer[layer])
        count = len(ablated_heads)
        candidates = [h for h in range(n_heads) if h not in ablated_heads]
        if len(candidates) < count:
            raise ValueError("Not enough candidate heads for control selection")
        pick = torch.randperm(len(candidates), generator=generator)[:count].tolist()
        control.extend((layer, candidates[i]) for i in pick)
    return sorted(control)


def zero_heads_hook(z, hook, heads):
    """Hook for `hook_z` that zeros the given head indices."""
    z[:, :, list(heads), :] = 0
    return z


def ablation_hooks(heads):
    """Group (layer, head) pairs into one zeroing hook per layer."""
    by_layer = defaultdict(list)
    for layer, head in heads:
        by_layer[layer].append(head)
    return [
        (f"blocks.{layer}.attn.hook_z", partial(zero_heads_hook, heads=sorted(hs)))
        for layer, hs in sorted(by_layer.items())
    ]


def score_heads(net, tokens, length, device="cpu"):
    """Per-sequence loss, induction score, and previous-token score for every head.

    Returns (losses (samples, 2 * length), induction (samples, layers, heads),
    previous (samples, layers, heads)).
    """
    names = [f"blocks.{layer}.attn.hook_pattern" for layer in range(net.cfg.n_layers)]
    ind_offset = induction_offset(length)
    losses, induction, previous = [], [], []
    with torch.inference_mode():
        for row in tokens:
            loss, cache = net.run_with_cache(
                row[None].to(device),
                names_filter=names,
                return_type="loss",
                loss_per_token=True,
            )
            losses.append(loss.cpu())
            induction.append(
                torch.stack([offset_score(cache[name].cpu(), length, ind_offset) for name in names])
            )
            previous.append(
                torch.stack(
                    [offset_score(cache[name].cpu(), length, PREVIOUS_OFFSET) for name in names]
                )
            )
    return torch.cat(losses), torch.stack(induction), torch.stack(previous)


def run(
    length=32,
    samples=16,
    seed=42,
    device="cpu",
    model="attn-only-2l",
    top_k=2,
    output=ROOT / "results/heads",
):
    if length < 2 or samples < 1 or top_k < 1:
        raise ValueError("Need length >= 2, samples >= 1, and top_k >= 1")

    net, pool, bos = load_model(model, device, min_ctx=1 + 2 * length)
    import matplotlib.pyplot as plt

    repeated, control = make_inputs(pool, bos, length, samples, seed)

    repeated_loss, ind_seq, prev_seq = score_heads(net, repeated, length, device)
    control_loss = per_token_losses(net, control, device)

    induction = ind_seq.mean(0)
    previous = prev_seq.mean(0)
    n_layers, n_heads = induction.shape
    if top_k > n_layers * n_heads:
        raise ValueError("top_k exceeds the number of heads in the model")
    ranked = rank_heads(induction)
    top = ranked[:top_k]
    ctrl = select_control_heads(top, net.cfg.n_heads, seed)

    def scores_for(heads):
        return [
            {
                "layer": layer,
                "head": head,
                "induction_score": induction[layer, head].item(),
                "previous_token_score": previous[layer, head].item(),
            }
            for layer, head in heads
        ]

    conditions = {
        "baseline": (repeated_loss, control_loss, []),
        "induction_ablated": (
            per_token_losses(net, repeated, device, ablation_hooks(top)),
            per_token_losses(net, control, device, ablation_hooks(top)),
            top,
        ),
        "control_ablated": (
            per_token_losses(net, repeated, device, ablation_hooks(ctrl)),
            per_token_losses(net, control, device, ablation_hooks(ctrl)),
            ctrl,
        ),
    }
    ablations = {}
    condition_losses = {}
    for name, (rep_loss, ctl_loss, heads) in conditions.items():
        condition_losses[name] = (rep_loss, ctl_loss)
        ablations[name] = {
            **summarize(rep_loss, ctl_loss, length),
            "heads": scores_for(heads),
        }

    summary = {
        "model": model,
        "device": device,
        "seed": seed,
        "length": length,
        "samples": samples,
        "top_k": top_k,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ablation": "zero",
        "induction_scores": induction.tolist(),
        "previous_token_scores": previous.tolist(),
        "induction_score_standard_error": (
            (ind_seq.std(0) / samples**0.5).tolist() if samples > 1 else None
        ),
        "ranked_heads": scores_for(ranked),
        "ablations": ablations,
        "torch_version": torch.__version__,
    }

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "repeated": repeated,
            "control": control,
            "induction_per_sequence": ind_seq,
            "previous_per_sequence": prev_seq,
            **{
                key: loss
                for name, (rep_loss, ctl_loss) in condition_losses.items()
                for key, loss in (
                    (f"{name}_repeated_loss", rep_loss),
                    (f"{name}_control_loss", ctl_loss),
                )
            },
        },
        output / "data.pt",
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, scores, title in (
        (axes[0], induction, "Induction score"),
        (axes[1], previous, "Previous-token score"),
    ):
        im = ax.imshow(scores.numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        for layer in range(n_layers):
            for head in range(n_heads):
                ax.text(
                    head,
                    layer,
                    f"{scores[layer, head].item():.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )
        fig.colorbar(im, ax=ax)

    names = list(ablations)
    benefits = [ablations[name]["paired_copying_benefit_nats"] for name in names]
    errors = [ablations[name]["paired_benefit_standard_error"] for name in names]
    axes[2].bar(names, benefits, yerr=errors)
    axes[2].set_ylabel("Paired copying benefit (nats)")
    fig.tight_layout()
    fig.savefig(output / "heads.png", dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--model", default="attn-only-2l")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/heads",
        help="Directory for summary.json, data.pt, heads.png (default: results/heads/)",
    )
    args = parser.parse_args()
    run(**vars(args))
