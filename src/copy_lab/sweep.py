"""Replicate the induction-head ablation across random seeds and block lengths."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch

from copy_lab.common import ROOT, load_model, per_token_losses
from copy_lab.experiment import make_inputs, summarize
from copy_lab.heads import (
    CONDITIONS,
    ablation_conditions,
    head_records,
    rank_heads,
    score_heads,
    select_control_heads,
)


def head_key(layer, head):
    return f"L{layer}H{head}"


def cell_key(length, seed):
    return f"L{length}_s{seed}"


def validate_grid(lengths, seeds, samples, top_k):
    if not lengths or any(length < 2 for length in lengths):
        raise ValueError("Need at least one length, and every length >= 2")
    if len(set(lengths)) != len(lengths):
        raise ValueError("Lengths must be unique")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Need at least one seed, and seeds must be unique")
    if samples < 1 or top_k < 1:
        raise ValueError("Need samples >= 1 and top_k >= 1")


def mean_std(values):
    """Mean and sample standard deviation (ddof=1); std is None for fewer than two values."""
    values = list(values)
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) >= 2 else None,
    }


def run_cell(net, pool, bos, length, seed, samples, top_k, device):
    """One (length, seed) cell; returns (JSON-ready record, tensors for data.pt)."""
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

    conditions = ablation_conditions(
        net, repeated, control, device, top, ctrl, (repeated_loss, control_loss)
    )
    summaries = {
        name: summarize(rep_loss, ctl_loss, length)
        for name, (rep_loss, ctl_loss, _heads) in conditions.items()
    }
    record = {
        "length": length,
        "seed": seed,
        "conditions": {
            name: {k: v for k, v in s.items() if k != "per_sequence_benefit"}
            for name, s in summaries.items()
        },
        "top_heads": head_records(top, induction, previous),
        "control_heads": head_records(ctrl, induction, previous),
        "induction_scores": induction.tolist(),
    }
    tensors = {
        "benefits": {
            name: torch.tensor(s["per_sequence_benefit"]) for name, s in summaries.items()
        },
        "induction_scores": induction,
        "previous_token_scores": previous,
        "induction_per_sequence": ind_seq,
    }
    return record, tensors


def aggregate(cells):
    """Per-length mean/std across seeds, sorted by length."""
    by_length = {}
    for cell in cells:
        by_length.setdefault(cell["length"], []).append(cell)
    per_length = []
    for length in sorted(by_length):
        group = by_length[length]
        n_seeds = len(group)
        counts = Counter(
            head_key(c["top_heads"][0]["layer"], c["top_heads"][0]["head"]) for c in group
        )
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        per_length.append(
            {
                "length": length,
                "n_seeds": n_seeds,
                "benefit": {
                    name: mean_std(
                        c["conditions"][name]["paired_copying_benefit_nats"] for c in group
                    )
                    for name in CONDITIONS
                },
                "top1_head_counts": dict(ordered),
                "top1_consistency": ordered[0][1] / n_seeds,
                "top1_induction_score": mean_std(
                    c["top_heads"][0]["induction_score"] for c in group
                ),
            }
        )
    return per_length


def plot_sweep(per_length, path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    lengths = [row["length"] for row in per_length]
    n_seeds = per_length[0]["n_seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in CONDITIONS:
        stats = [row["benefit"][name] for row in per_length]
        axes[0].errorbar(
            lengths,
            [s["mean"] for s in stats],
            yerr=[s["std"] or 0.0 for s in stats],
            marker="o",
            capsize=3,
            label=name,
        )
    stats = [row["top1_induction_score"] for row in per_length]
    axes[1].errorbar(
        lengths,
        [s["mean"] for s in stats],
        yerr=[s["std"] or 0.0 for s in stats],
        marker="o",
        capsize=3,
    )
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Tokens per block")
    axes[0].set_ylabel("Paired copying benefit (nats)")
    axes[0].set_title(f"Copying benefit vs length (mean ± std over {n_seeds} seeds)")
    axes[0].legend()
    axes[1].set_ylabel("Top-1 head induction score")
    axes[1].set_title("Top-1 induction score vs length")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(
    lengths=(8, 16, 32, 64),
    seeds=(0, 1, 2, 3, 4),
    samples=16,
    top_k=2,
    device="cpu",
    model="attn-only-2l",
    output=ROOT / "results/sweep",
):
    """Run the grid, write summary.json, data.pt and sweep.png, and return the full summary.

    Prints only the per-length aggregate, not the (much longer) per-cell records.
    """
    lengths, seeds = list(lengths), list(seeds)
    validate_grid(lengths, seeds, samples, top_k)

    net, pool, bos = load_model(model, device, min_ctx=1 + 2 * max(lengths))

    cells, tensors = [], {}
    for length in lengths:
        for seed in seeds:
            record, cell_tensors = run_cell(net, pool, bos, length, seed, samples, top_k, device)
            cells.append(record)
            tensors[cell_key(length, seed)] = cell_tensors
            losses = " ".join(
                f"{name}={record['conditions'][name]['paired_copying_benefit_nats']:.2f}"
                for name in CONDITIONS
            )
            print(f"L={length} seed={seed} {losses}", flush=True)

    per_length = aggregate(cells)
    summary = {
        "model": model,
        "device": device,
        "lengths": lengths,
        "seeds": seeds,
        "samples": samples,
        "top_k": top_k,
        "n_layers": net.cfg.n_layers,
        "n_heads": net.cfg.n_heads,
        "ablation": "zero",
        "conditions": list(CONDITIONS),
        "cells": cells,
        "per_length": per_length,
        "torch_version": torch.__version__,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    torch.save(tensors, output / "data.pt")
    plot_sweep(per_length, output / "sweep.png")
    print(json.dumps(per_length, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--model", default="attn-only-2l")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/sweep",
        help="Directory for summary.json, data.pt, sweep.png (default: results/sweep/)",
    )
    args = parser.parse_args()
    run(**vars(args))
