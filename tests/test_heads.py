import pytest
import torch

from copy_lab.heads import (
    ablation_hooks,
    induction_offset,
    offset_score,
    rank_heads,
    run,
    select_control_heads,
    zero_heads_hook,
)


def test_offset_score_extracts_induction_and_previous_cells():
    # length=3 -> n_pos=7, scored queries (block_slices(3)[1]) are 4, 5.
    patterns = torch.zeros(1, 2, 7, 7)
    # Head 0: induction cells (offset 1-3=-2): (4,2) and (5,3).
    patterns[0, 0, 4, 2] = 1.0
    patterns[0, 0, 5, 3] = 1.0
    # Head 1: previous-token cells (offset -1): (4,3) and (5,4).
    patterns[0, 1, 4, 3] = 1.0
    patterns[0, 1, 5, 4] = 1.0

    assert offset_score(patterns, 3, induction_offset(3)).tolist() == [1.0, 0.0]
    assert offset_score(patterns, 3, -1).tolist() == [0.0, 1.0]

    # Add a second batch element with the same cells at 0.5 to check the batch mean.
    patterns2 = torch.zeros(2, 2, 7, 7)
    patterns2[0] = patterns[0]
    patterns2[1, 0, 4, 2] = 0.5
    patterns2[1, 0, 5, 3] = 0.5
    patterns2[1, 1, 4, 3] = 0.5
    patterns2[1, 1, 5, 4] = 0.5
    assert offset_score(patterns2, 3, induction_offset(3)).tolist() == pytest.approx([0.75, 0.0])
    assert offset_score(patterns2, 3, -1).tolist() == pytest.approx([0.0, 0.75])


def test_offset_score_rejects_bad_shapes_and_offsets():
    with pytest.raises(ValueError):
        offset_score(torch.zeros(1, 1, 3, 3), 1, 0)  # length < 2
    with pytest.raises(ValueError):
        offset_score(torch.zeros(1, 1, 5, 5), 3, -2)  # shape mismatch: n_pos should be 7
    with pytest.raises(ValueError):
        offset_score(torch.zeros(1, 1, 7, 7), 3, 5)  # key index out of range


def test_rank_heads_sorts_descending():
    # Deviation from the plan's stated expected value: the plan's example
    # (`rank_heads([[0.1, 0.9], [0.5, 0.2]]) == [(0,1), (1,0), (0,0), (1,1)]`)
    # is not a descending sort of the given scores (0.1 precedes 0.2). A plain
    # descending sort of {(0,0): 0.1, (0,1): 0.9, (1,0): 0.5, (1,1): 0.2} is
    # [(0,1), (1,0), (1,1), (0,0)], which is what this test asserts.
    assert rank_heads([[0.1, 0.9], [0.5, 0.2]]) == [(0, 1), (1, 0), (1, 1), (0, 0)]


def test_select_control_heads_disjoint_reproducible_and_validated():
    controls = select_control_heads([(1, 4), (1, 6)], 8, seed=7)
    assert len(controls) == 2
    assert all(layer == 1 for layer, _ in controls)
    heads = {head for _, head in controls}
    assert heads.isdisjoint({4, 6})
    assert controls == sorted(controls)

    assert select_control_heads([(1, 4), (1, 6)], 8, seed=7) == controls

    with pytest.raises(ValueError):
        select_control_heads([(1, 4), (1, 6)], 1, seed=7)


def test_zero_heads_hook_zeros_only_given_heads():
    z = torch.ones(1, 5, 4, 3)
    out = zero_heads_hook(z, hook=None, heads=[1, 3])
    for head in range(4):
        expected = 0.0 if head in (1, 3) else 1.0
        assert torch.all(out[:, :, head, :] == expected)


def test_ablation_hooks_groups_by_layer():
    hooks = ablation_hooks([(1, 4), (0, 2), (1, 6)])
    names = [name for name, _ in hooks]
    assert names == ["blocks.0.attn.hook_z", "blocks.1.attn.hook_z"]

    layer1_fn = dict(hooks)["blocks.1.attn.hook_z"]
    z = torch.ones(1, 3, 8, 2)
    out = layer1_fn(z, hook=None)
    for head in range(8):
        expected = 0.0 if head in (4, 6) else 1.0
        assert torch.all(out[:, :, head, :] == expected)


def test_run_end_to_end_with_fake_model(tmp_path, patch_from_pretrained):
    summary = run(length=4, samples=3, seed=1, top_k=1, output=tmp_path)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "data.pt").exists()
    assert (tmp_path / "heads.png").exists()

    ranked0 = summary["ranked_heads"][0]
    assert (ranked0["layer"], ranked0["head"]) == (1, 4)
    assert summary["previous_token_scores"][0][7] == 1.0

    controls = summary["ablations"]["control_ablated"]["heads"]
    assert all(h["layer"] == 1 for h in controls)
    assert {h["head"] for h in controls}.isdisjoint({4})

    baseline_benefit = summary["ablations"]["baseline"]["paired_copying_benefit_nats"]
    ablated_benefit = summary["ablations"]["induction_ablated"]["paired_copying_benefit_nats"]
    assert ablated_benefit < baseline_benefit

    data = torch.load(tmp_path / "data.pt")
    assert data["repeated"].shape == (3, 9)
    assert data["induction_per_sequence"].shape == (3, 2, 8)


def test_run_validates_before_loading(tmp_path, forbid_from_pretrained):
    with pytest.raises(ValueError):
        run(length=1, output=tmp_path)
    with pytest.raises(ValueError):
        run(top_k=0, output=tmp_path)
