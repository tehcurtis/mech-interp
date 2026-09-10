import matplotlib

matplotlib.use("Agg")

import pytest
import torch

from copy_lab.experiment import block_slices, make_inputs, region_losses, run, summarize


def test_matched_targets_and_reproducibility():
    repeated, control = make_inputs(torch.arange(1, 100), 0, length=8)
    assert torch.equal(repeated[:, 1:9], repeated[:, 9:])
    assert torch.equal(repeated[:, 9:], control[:, 9:])
    assert not torch.equal(repeated[:, 1:9], control[:, 1:9])
    assert torch.equal(repeated, make_inputs(torch.arange(1, 100), 0, length=8)[0])


def test_loss_alignment_excludes_both_boundary_targets():
    # For BOS a b c a b c, loss indices 1,2 and 4,5 predict b,c.
    first, second = region_losses(torch.tensor([[999.0, 2.0, 4.0, 999.0, 6.0, 8.0]]), 3)
    assert first.item() == 3
    assert second.item() == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 1},
        {"samples": 0},
        {"pool": torch.arange(1, 2)},
    ],
)
def test_make_inputs_rejects_bad_arguments(kwargs):
    args = {"pool": torch.arange(1, 100), "bos": 0}
    args.update(kwargs)
    with pytest.raises(ValueError):
        make_inputs(**args)


def test_region_losses_rejects_bad_shapes():
    with pytest.raises(ValueError):
        region_losses(torch.zeros(2, 4), 1)
    with pytest.raises(ValueError):
        region_losses(torch.zeros(2, 5), 3)


def test_block_slices_match_layout():
    assert block_slices(3) == (slice(1, 3), slice(4, 6))


def test_summarize_uses_control_minus_repeated():
    # length=3 -> shape (samples, 6); block_slices(3) = (slice(1,3), slice(4,6))
    repeated_loss = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 2.0, 2.0],
            [0.0, 1.0, 1.0, 0.0, 4.0, 4.0],
        ]
    )
    control_loss = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 5.0, 5.0],
            [0.0, 1.0, 1.0, 0.0, 7.0, 7.0],
        ]
    )
    result = summarize(repeated_loss, control_loss, 3)
    repeated_second_mean = torch.tensor([2.0, 4.0]).mean().item()
    control_second_mean = torch.tensor([5.0, 7.0]).mean().item()
    assert result["paired_copying_benefit_nats"] == pytest.approx(
        control_second_mean - repeated_second_mean
    )
    assert len(result["per_sequence_benefit"]) == 2


def test_run_end_to_end_with_fake_model(tmp_path, monkeypatch):
    class FakeConfig:
        n_ctx = 64
        d_vocab = 50

    class FakeTokenizer:
        bos_token_id = 0
        all_special_ids = [0, 1]

    class FakeModel:
        cfg = FakeConfig()
        tokenizer = FakeTokenizer()

        def eval(self):
            pass

        def __call__(self, tokens, return_type, loss_per_token):
            n = tokens.shape[1] - 1
            return torch.arange(n, dtype=torch.float)[None]

    def fake_from_pretrained(name, device="cpu"):
        return FakeModel()

    import transformer_lens

    monkeypatch.setattr(transformer_lens.HookedTransformer, "from_pretrained", fake_from_pretrained)

    summary = run(length=4, samples=3, seed=1, output=tmp_path)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "data.pt").exists()
    assert (tmp_path / "copying.png").exists()
    assert summary["model"] == "attn-only-2l"
    assert summary["samples"] == 3
    data = torch.load(tmp_path / "data.pt")
    assert data["repeated"].shape == (3, 9)

    def unreachable_from_pretrained(name, device="cpu"):
        raise AssertionError("from_pretrained should not be called")

    monkeypatch.setattr(
        transformer_lens.HookedTransformer, "from_pretrained", unreachable_from_pretrained
    )
    with pytest.raises(ValueError):
        run(length=1, output=tmp_path)
