import matplotlib

matplotlib.use("Agg")

import pytest
import torch


class FakeConfig:
    n_layers = 2
    n_heads = 8
    n_ctx = 256
    d_vocab = 50


class FakeTokenizer:
    bos_token_id = 0
    all_special_ids = [0, 1]


class FakeModel:
    cfg = FakeConfig()
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def eval(self):
        pass

    def __call__(self, tokens, return_type, loss_per_token):
        n = tokens.shape[1] - 1
        return torch.arange(n, dtype=torch.float)[None]

    def run_with_cache(self, tokens, names_filter, **kw):
        n_pos = tokens.shape[1]
        length = (n_pos - 1) // 2
        second = range(length + 1, 2 * length)
        patterns = {}
        for layer in range(self.cfg.n_layers):
            p = torch.zeros(1, self.cfg.n_heads, n_pos, n_pos)
            if layer == 1:
                for q in second:
                    p[0, 4, q, q - length + 1] = 1.0  # induction cell
            if layer == 0:
                for q in second:
                    p[0, 7, q, q - 1] = 1.0  # previous-token cell
            patterns[f"blocks.{layer}.attn.hook_pattern"] = p
        loss = self.__call__(tokens, **kw)
        return loss, patterns

    def run_with_hooks(self, tokens, fwd_hooks, **kw):
        n_pos = tokens.shape[1]
        length = (n_pos - 1) // 2
        zeroed = 0
        for _name, fn in fwd_hooks:
            z = torch.ones(1, n_pos, self.cfg.n_heads, 4)
            z = fn(z, hook=None)
            zeroed += sum(1 for h in range(self.cfg.n_heads) if torch.all(z[:, :, h, :] == 0))
        loss = self.__call__(tokens, **kw)
        is_repeated = torch.equal(tokens[0, 1 : length + 1], tokens[0, length + 1 :])
        if is_repeated:
            loss = loss + zeroed
        return loss


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def patch_from_pretrained(monkeypatch, fake_model):
    """Route HookedTransformer.from_pretrained to the fake; records (name, device) in calls."""
    import transformer_lens

    def fake_from_pretrained(name, device="cpu"):
        fake_model.calls.append((name, device))
        return fake_model

    monkeypatch.setattr(transformer_lens.HookedTransformer, "from_pretrained", fake_from_pretrained)
    return fake_model


@pytest.fixture
def forbid_from_pretrained(monkeypatch):
    """from_pretrained raises AssertionError; for argument-validation tests."""
    import transformer_lens

    def unreachable_from_pretrained(name, device="cpu"):
        raise AssertionError("from_pretrained should not be called")

    monkeypatch.setattr(
        transformer_lens.HookedTransformer, "from_pretrained", unreachable_from_pretrained
    )
