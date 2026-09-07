import torch
from copy_lab.experiment import make_inputs, region_losses


def test_matched_targets_and_reproducibility():
    repeated, control = make_inputs(torch.arange(1, 100), 0, length=8)
    assert torch.equal(repeated[:, 1:9], repeated[:, 9:])
    assert torch.equal(repeated[:, 9:], control[:, 9:])
    assert not torch.equal(repeated[:, 1:9], control[:, 1:9])
    assert torch.equal(repeated, make_inputs(torch.arange(1, 100), 0, length=8)[0])


def test_loss_alignment_excludes_both_boundary_targets():
    # For BOS a b c a b c, loss indices 1,2 and 4,5 predict b,c.
    first, second = region_losses(torch.tensor([[999., 2., 4., 999., 6., 8.]]), 3)
    assert first.item() == 3
    assert second.item() == 7
