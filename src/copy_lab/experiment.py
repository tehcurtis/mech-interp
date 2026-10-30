"""Compare BOS+A+A with BOS+B+A: same targets, different earlier context."""

import argparse
import json
from pathlib import Path

import torch

from copy_lab.common import ROOT, load_model, per_token_losses


def make_inputs(pool, bos, length=32, samples=16, seed=42):
    """Random token IDs, excluding special tokens; accidental repeats are allowed."""
    if length < 2 or samples < 1 or len(pool) < 2:
        raise ValueError("Need length >= 2, samples >= 1, and at least two tokens")
    rng = torch.Generator().manual_seed(seed)
    a = pool[torch.randint(len(pool), (samples, length), generator=rng)]
    b = pool[torch.randint(len(pool), (samples, length), generator=rng)]
    prefix = torch.full((samples, 1), bos, dtype=torch.long)
    return torch.cat((prefix, a, a), 1), torch.cat((prefix, b, a), 1)


def block_slices(length):
    """Loss-index slices for the scored targets in block 1 and block 2.

    Sequences are laid out [BOS, block1(length), block2(length)]. Loss index i
    scores target token i+1, so the first target of each block is skipped.
    """
    return slice(1, length), slice(length + 1, 2 * length)


def region_losses(losses, length):
    if length < 2:
        raise ValueError("Need length >= 2")
    if losses.shape[1] != 2 * length:
        raise ValueError(
            f"Expected {2 * length} loss positions for length={length}, got {losses.shape[1]}"
        )
    first_slice, second_slice = block_slices(length)
    return losses[:, first_slice].mean(1), losses[:, second_slice].mean(1)


def summarize(repeated_loss, control_loss, length):
    first, second = region_losses(repeated_loss, length)
    _, baseline = region_losses(control_loss, length)
    benefit = baseline - second
    samples = benefit.shape[0]
    return {
        "first_block_loss": first.mean().item(),
        "repeated_second_block_loss": second.mean().item(),
        "control_second_block_loss": baseline.mean().item(),
        "paired_copying_benefit_nats": benefit.mean().item(),
        "paired_benefit_standard_error": benefit.std().item() / samples**0.5
        if samples > 1
        else None,
        "per_sequence_benefit": benefit.tolist(),
    }


def run(
    length=32,
    samples=16,
    seed=42,
    device="cpu",
    model="attn-only-2l",
    output=ROOT / "results",
):
    if length < 2 or samples < 1:
        raise ValueError("Need length >= 2 and samples >= 1")

    net, pool, bos = load_model(model, device, min_ctx=1 + 2 * length)
    import matplotlib.pyplot as plt

    repeated, control = make_inputs(pool, bos, length, samples, seed)

    # Process one sequence at a time to bound memory from full-vocabulary logits.
    repeated_loss = per_token_losses(net, repeated, device)
    control_loss = per_token_losses(net, control, device)
    summary = {
        "model": model,
        "device": device,
        "seed": seed,
        "length": length,
        "samples": samples,
        **summarize(repeated_loss, control_loss, length),
        "torch_version": torch.__version__,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "repeated": repeated,
            "control": control,
            "repeated_loss": repeated_loss,
            "control_loss": control_loss,
        },
        output / "data.pt",
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = range(2, length + 1)
    _, second_slice = block_slices(length)
    ax.plot(
        positions,
        repeated_loss[:, second_slice].mean(0),
        label="Repeated context: A → A",
    )
    ax.plot(
        positions,
        control_loss[:, second_slice].mean(0),
        label="Unrelated context: B → A",
    )
    ax.set(
        xlabel="Target position within second block",
        ylabel="Prediction loss (nats; lower is better)",
        title="Does earlier exposure help a transformer copy?",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "copying.png", dpi=160)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results",
        help="Directory for summary.json, data.pt, copying.png (default: results/)",
    )
    args = parser.parse_args()
    run(**vars(args))
