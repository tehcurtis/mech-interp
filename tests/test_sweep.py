import json
import math
import statistics

import pytest
import torch

from copy_lab.heads import CONDITIONS
from copy_lab.sweep import aggregate, mean_std, run, validate_grid


def make_cell(length, seed, benefits, top=(1, 6), score=0.5):
    """A minimal cell record in the layout produced by run_cell."""
    conditions = {
        name: {"paired_copying_benefit_nats": benefit}
        for name, benefit in zip(CONDITIONS, benefits, strict=True)
    }
    layer, head = top
    return {
        "length": length,
        "seed": seed,
        "conditions": conditions,
        "top_heads": [{"layer": layer, "head": head, "induction_score": score}],
    }


INVALID_GRIDS = [
    {"lengths": [1]},
    {"lengths": [8, 8]},
    {"seeds": []},
    {"samples": 0},
    {"top_k": 0},
]


def test_aggregate_mean_std_and_ordering():
    baselines = {16: [10.0, 11.0, 15.0], 8: [4.0, 5.0, 9.0]}
    cells = [
        make_cell(length, seed, (b, b / 10, b + 1))
        for length, values in baselines.items()
        for seed, b in enumerate(values)
    ]
    per_length = aggregate(cells)

    assert [row["length"] for row in per_length] == [8, 16]
    for row in per_length:
        values = baselines[row["length"]]
        assert row["n_seeds"] == 3
        assert row["benefit"]["baseline"]["mean"] == pytest.approx(statistics.mean(values))
        assert row["benefit"]["baseline"]["std"] == pytest.approx(statistics.stdev(values))
        shifted = [v + 1 for v in values]
        assert row["benefit"]["control_ablated"]["mean"] == pytest.approx(statistics.mean(shifted))
        assert row["benefit"]["control_ablated"]["std"] == pytest.approx(statistics.stdev(shifted))


def test_aggregate_top1_counts():
    cells = [
        make_cell(8, 0, (1, 0, 1), top=(1, 6), score=0.6),
        make_cell(8, 1, (1, 0, 1), top=(1, 6), score=0.9),
        make_cell(8, 2, (1, 0, 1), top=(1, 7), score=0.3),
    ]
    (row,) = aggregate(cells)

    assert row["top1_head_counts"] == {"L1H6": 2, "L1H7": 1}
    assert list(row["top1_head_counts"]) == ["L1H6", "L1H7"]
    assert row["top1_consistency"] == pytest.approx(2 / 3)
    assert row["top1_induction_score"]["mean"] == pytest.approx(statistics.mean([0.6, 0.9, 0.3]))


def test_aggregate_single_seed_std_is_none():
    (row,) = aggregate([make_cell(8, 0, (1.0, 0.0, 1.0))])

    assert row["n_seeds"] == 1
    assert all(stats["std"] is None for stats in row["benefit"].values())
    assert row["top1_induction_score"]["std"] is None
    json.dumps(row, allow_nan=False)


def test_mean_std():
    assert mean_std([2.0, 4.0]) == {"mean": 3.0, "std": pytest.approx(math.sqrt(2))}
    assert mean_std([5.0]) == {"mean": 5.0, "std": None}
    assert mean_std([3.0, 3.0, 3.0]) == {"mean": 3.0, "std": 0.0}


@pytest.mark.parametrize("override", INVALID_GRIDS)
def test_validate_grid(override):
    grid = {"lengths": [8, 16], "seeds": [0, 1], "samples": 4, "top_k": 2}
    validate_grid(**grid)
    with pytest.raises(ValueError):
        validate_grid(**{**grid, **override})


def test_run_end_to_end_with_fake_model(tmp_path, patch_from_pretrained):
    summary = run(lengths=[4, 8], seeds=[0, 1], samples=3, top_k=1, output=tmp_path)

    for name in ("summary.json", "data.pt", "sweep.png"):
        assert (tmp_path / name).exists()
    assert len(patch_from_pretrained.calls) == 1
    assert len(summary["cells"]) == 4

    data = torch.load(tmp_path / "data.pt")
    assert set(data) == {"L4_s0", "L4_s1", "L8_s0", "L8_s1"}
    for tensors in data.values():
        assert set(tensors["benefits"]) == set(CONDITIONS)
        assert all(b.shape == (3,) for b in tensors["benefits"].values())
        assert tensors["induction_scores"].shape == (2, 8)
        assert tensors["previous_token_scores"].shape == (2, 8)
        assert tensors["induction_per_sequence"].shape == (3, 2, 8)

    for cell in summary["cells"]:
        top = cell["top_heads"][0]
        assert (top["layer"], top["head"]) == (1, 4)

    assert [row["length"] for row in summary["per_length"]] == [4, 8]
    for row in summary["per_length"]:
        assert row["n_seeds"] == 2
        assert row["top1_head_counts"] == {"L1H4": 2}
        assert row["top1_consistency"] == 1.0
        benefit = row["benefit"]
        assert benefit["induction_ablated"]["mean"] < benefit["baseline"]["mean"]
        for stats in [*benefit.values(), row["top1_induction_score"]]:
            assert math.isfinite(stats["mean"])
            assert math.isfinite(stats["std"])

    assert json.loads((tmp_path / "summary.json").read_text()) == summary


@pytest.mark.parametrize("override", INVALID_GRIDS)
def test_run_validates_before_loading(tmp_path, forbid_from_pretrained, override):
    grid = {"lengths": [4, 8], "seeds": [0, 1], "samples": 3, "top_k": 1}
    with pytest.raises(ValueError):
        run(**{**grid, **override}, output=tmp_path)


def test_run_single_seed(tmp_path, patch_from_pretrained):
    summary = run(lengths=[4], seeds=[0], samples=3, top_k=1, output=tmp_path)

    (row,) = summary["per_length"]
    assert row["n_seeds"] == 1
    assert all(stats["std"] is None for stats in row["benefit"].values())
    assert (tmp_path / "sweep.png").exists()
