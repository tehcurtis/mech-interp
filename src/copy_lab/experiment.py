"""Compare BOS+A+A with BOS+B+A: same targets, different earlier context."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import torch


def make_inputs(pool, bos, length=32, samples=16, seed=42):
    """Random token IDs, excluding special tokens; accidental repeats are allowed."""
    if length < 2 or samples < 1 or len(pool) < 2:
        raise ValueError("Need length >= 2, samples >= 1, and at least two tokens")
    rng = torch.Generator().manual_seed(seed)
    a = pool[torch.randint(len(pool), (samples, length), generator=rng)]
    b = pool[torch.randint(len(pool), (samples, length), generator=rng)]
    prefix = torch.full((samples, 1), bos, dtype=torch.long)
    return torch.cat((prefix, a, a), 1), torch.cat((prefix, b, a), 1)


def region_losses(losses, length):
    # Loss index i scores target token i+1. Skip the first token of each block.
    return losses[:, 1:length].mean(1), losses[:, length + 1:2 * length].mean(1)


def run(length=32, samples=16, seed=42, device="cpu", output=ROOT / "results"):
    from transformer_lens import HookedTransformer
    import matplotlib.pyplot as plt
    model = HookedTransformer.from_pretrained("attn-only-2l", device=device)
    model.eval()
    if 1 + 2 * length > model.cfg.n_ctx:
        raise ValueError("Sequence exceeds the model's context window")
    special = set(model.tokenizer.all_special_ids)
    pool = torch.tensor([i for i in range(model.cfg.d_vocab) if i not in special])
    bos = model.tokenizer.bos_token_id
    if bos is None:
        raise ValueError("This experiment requires a BOS token")
    repeated, control = make_inputs(pool, bos, length, samples, seed)
    # Process one sequence at a time to bound memory from full-vocabulary logits.
    def evaluate(tokens):
        with torch.inference_mode():
            return torch.cat([model(row[None].to(device), return_type="loss", loss_per_token=True).cpu() for row in tokens])
    repeated_loss, control_loss = evaluate(repeated), evaluate(control)
    first, second = region_losses(repeated_loss, length)
    _, baseline = region_losses(control_loss, length)
    benefit = baseline - second
    summary = {
        "model": "attn-only-2l", "device": device, "seed": seed,
        "length": length, "samples": samples,
        "first_block_loss": first.mean().item(),
        "repeated_second_block_loss": second.mean().item(),
        "control_second_block_loss": baseline.mean().item(),
        "paired_copying_benefit_nats": benefit.mean().item(),
        "paired_benefit_standard_error": benefit.std().item() / samples**0.5 if samples > 1 else None,
        "per_sequence_benefit": benefit.tolist(),
        "torch_version": torch.__version__,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"repeated": repeated, "control": control, "repeated_loss": repeated_loss, "control_loss": control_loss}, output / "data.pt")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = range(2, length + 1)
    ax.plot(positions, repeated_loss[:, length + 1:2 * length].mean(0), label="Repeated context: A → A")
    ax.plot(positions, control_loss[:, length + 1:2 * length].mean(0), label="Unrelated context: B → A")
    ax.set(xlabel="Target position within second block", ylabel="Prediction loss (nats; lower is better)", title="Does earlier exposure help a transformer copy?")
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
    args = parser.parse_args()
    run(**vars(args))
