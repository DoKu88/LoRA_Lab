"""Phase 2 Sprint 1 — hypernetwork plumbing invariants (CPU, no network).

Validates generate -> apply -> backprop on a synthetic tiny model:
shape contract, no-op-adapter-leaves-logits-unchanged, grads reach the
hypernetwork, conditioning wiring, and a reconstruction overfit step.
"""

import pytest
import torch
import torch.nn as nn

from lora_lab.hypernet.apply import LoRARegistry, lora_injected, target_specs
from lora_lab.hypernet.heads import HEADS, estimate_params
from lora_lab.hypernet.model import HyperLoRA, delta_w
from lora_lab.hypernet.recon import reconstruction_loss

torch.manual_seed(0)


class _TinyAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, x):
        return self.o(self.q_proj(x) + self.v_proj(x))


class _TinyModel(nn.Module):
    def __init__(self, d=24, n=2):
        super().__init__()
        self.layers = nn.ModuleList([_TinyAttn(d) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


TARGETS = ["q_proj", "v_proj"]
D_TASK = 8


def _setup(d=24, n=2):
    model = _TinyModel(d, n).eval()
    for p in model.parameters():
        p.requires_grad_(False)  # base frozen
    specs = target_specs(model, TARGETS)
    hyper = HyperLoRA(specs, d_task=D_TASK, r=4, alpha=8)
    return model, specs, hyper


def test_target_specs_shapes():
    model, specs, _ = _setup(d=24, n=2)
    # 2 layers × {q_proj, v_proj} = 4 targets, each 24->24
    assert len(specs) == 4
    assert all(v == (24, 24) for v in specs.values())


def test_generated_shapes():
    _, specs, hyper = _setup()
    adapter = hyper(torch.randn(D_TASK))
    assert set(adapter) == set(specs)
    for key, (a, b) in adapter.items():
        in_f, out_f = specs[key]
        assert a.shape == (4, in_f)
        assert b.shape == (out_f, 4)


def test_noop_adapter_leaves_logits_unchanged():
    """B0 is zero-init => ΔW = 0 => injected forward == base forward."""
    model, _, hyper = _setup()
    x = torch.randn(3, 5, 24)
    base_out = model(x)
    reg = LoRARegistry()
    reg.set_adapter(hyper(torch.randn(D_TASK)))
    with lora_injected(model, TARGETS, reg, scaling=hyper.scaling):
        inj_out = model(x)
    assert torch.allclose(base_out, inj_out, atol=1e-6)


def test_grads_reach_hypernetwork():
    model, _, hyper = _setup()
    x = torch.randn(2, 4, 24)
    reg = LoRARegistry()
    reg.set_adapter(hyper(torch.randn(D_TASK)))
    with lora_injected(model, TARGETS, reg, scaling=hyper.scaling):
        out = model(x)
    out.pow(2).mean().backward()
    grads = [p.grad for p in hyper.parameters() if p.grad is not None]
    assert grads, "no hypernetwork parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads), "all hypernetwork grads are zero"


def test_conditioning_mechanism():
    """With a non-degenerate gate + B0, two task embeddings -> different ΔW."""
    _, specs, hyper = _setup()
    with torch.no_grad():
        for p in hyper.B0.values():
            p.copy_(torch.randn_like(p) * 0.1)
        hyper.gate.weight.copy_(torch.randn_like(hyper.gate.weight) * 0.5)
    a1 = hyper(torch.randn(D_TASK))
    a2 = hyper(torch.randn(D_TASK))
    k = next(iter(specs))
    dw1 = delta_w(*a1[k], hyper.scaling)
    dw2 = delta_w(*a2[k], hyper.scaling)
    assert not torch.allclose(dw1, dw2, atol=1e-4)


def test_reconstruction_overfit_decreases():
    """Full generate -> recon-loss -> backprop -> update loop reduces the loss."""
    _, specs, hyper = _setup()
    torch.manual_seed(1)
    target = {k: (torch.randn(4, specs[k][0]), torch.randn(specs[k][1], 4)) for k in specs}
    task_emb = torch.randn(D_TASK)
    opt = torch.optim.Adam(hyper.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = reconstruction_loss(hyper(task_emb), target, scaling=hyper.scaling)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], f"recon loss did not drop: {losses[0]:.3f}->{losses[-1]:.3f}"


# ---- Sprint 2: output-parameterization heads ------------------------------
@pytest.mark.parametrize("name", ["full", "lowrank", "vera"])
def test_head_shapes_and_noop(name):
    d_cond, in_f, out_f, r = 32, 48, 40, 8
    head = HEADS[name](d_cond, in_f, out_f, r)
    a, b = head(torch.randn(d_cond))
    assert a.shape == (r, in_f) and b.shape == (out_f, r)
    # B path is zero-init for every parameterization => ΔW == 0 at init
    assert torch.allclose(b @ a, torch.zeros(out_f, in_f), atol=1e-6)


def test_head_param_budget_ordering():
    # A Mistral-like target set: q/k/v over a few layers (GQA: kv out smaller).
    specs = {}
    for layer in range(4):
        specs[f"l{layer}.q"] = (4096, 4096)
        specs[f"l{layer}.k"] = (4096, 1024)
        specs[f"l{layer}.v"] = (4096, 1024)
    d_cond, r = 64, 16
    vera = estimate_params("vera", specs, d_cond, r)
    lowrank = estimate_params("lowrank", specs, d_cond, r)
    full = estimate_params("full", specs, d_cond, r)
    # smallest-output ordering the S2 ladder assumes
    assert vera < lowrank < full


# ---- Sprint 5: retrieval baseline -----------------------------------------
class _FakeEncoder:
    """Deterministic encoder: each description -> a fixed vector by keyword."""
    dim = 3
    _VECS = {
        "sentiment": [1.0, 0.0, 0.0],
        "entailment": [0.0, 1.0, 0.0],
        "translation": [0.0, 0.0, 1.0],
    }

    def encode(self, descriptions):
        import torch
        rows = []
        for d in descriptions:
            v = [0.0, 0.0, 0.0]
            for kw, vec in self._VECS.items():
                if kw in d:
                    v = vec
            rows.append(v)
        return torch.tensor(rows)


def test_retrieval_nearest_and_train_only():
    from lora_lab.hypernet.retrieval import RetrievalIndex
    enc = _FakeEncoder()
    train = {
        "task_sent": {"description": "classify sentiment", "adapter_repo": "A/sent"},
        "task_nli": {"description": "decide entailment", "adapter_repo": "A/nli"},
    }
    idx = RetrievalIndex.build(train, enc)
    assert len(idx) == 2
    # a held-out entailment-style description retrieves the NLI train task
    hit = idx.query("textual entailment between two sentences", enc)[0]
    assert hit.task == "task_nli"
    assert hit.payload["adapter_repo"] == "A/nli"
    # the index contains only train tasks (held-out can't retrieve itself)
    assert "task_heldout" not in idx.tasks


def test_retrieval_scores_sorted():
    from lora_lab.hypernet.retrieval import RetrievalIndex
    enc = _FakeEncoder()
    train = {f"t{i}": {"description": d} for i, d in
             enumerate(["sentiment", "entailment", "translation"])}
    idx = RetrievalIndex.build(train, enc)
    res = idx.query("sentiment polarity", enc, k=3)
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)
