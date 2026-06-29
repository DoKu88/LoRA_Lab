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
TASK_DIM = 8


def _setup(d=24, n=2):
    model = _TinyModel(d, n).eval()
    for p in model.parameters():
        p.requires_grad_(False)  # base frozen
    specs = target_specs(model, TARGETS)
    hyper = HyperLoRA(specs, task_dim=TASK_DIM, rank=4, alpha=8)
    return model, specs, hyper


def test_target_specs_shapes():
    model, specs, _ = _setup(d=24, n=2)
    # 2 layers × {q_proj, v_proj} = 4 targets, each 24->24
    assert len(specs) == 4
    assert all(v == (24, 24) for v in specs.values())


def test_generated_shapes():
    _, specs, hyper = _setup()
    adapter = hyper(torch.randn(TASK_DIM))
    assert set(adapter) == set(specs)
    for key, (a, b) in adapter.items():
        in_f, out_f = specs[key]
        assert a.shape == (4, in_f)
        assert b.shape == (out_f, 4)


def test_noop_adapter_leaves_logits_unchanged():
    """base_b is zero-init => ΔW = 0 => injected forward == base forward."""
    model, _, hyper = _setup()
    x = torch.randn(3, 5, 24)
    base_out = model(x)
    reg = LoRARegistry()
    reg.set_adapter(hyper(torch.randn(TASK_DIM)))
    with lora_injected(model, TARGETS, reg, scaling=hyper.scaling):
        inj_out = model(x)
    assert torch.allclose(base_out, inj_out, atol=1e-6)


def test_grads_reach_hypernetwork():
    model, _, hyper = _setup()
    x = torch.randn(2, 4, 24)
    reg = LoRARegistry()
    reg.set_adapter(hyper(torch.randn(TASK_DIM)))
    with lora_injected(model, TARGETS, reg, scaling=hyper.scaling):
        out = model(x)
    out.pow(2).mean().backward()
    grads = [p.grad for p in hyper.parameters() if p.grad is not None]
    assert grads, "no hypernetwork parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads), "all hypernetwork grads are zero"


def test_conditioning_mechanism():
    """With a non-degenerate gate + base_b, two task embeddings -> different ΔW."""
    _, specs, hyper = _setup()
    with torch.no_grad():
        for p in hyper.base_b.values():
            p.copy_(torch.randn_like(p) * 0.1)
        hyper.gate.weight.copy_(torch.randn_like(hyper.gate.weight) * 0.5)
    a1 = hyper(torch.randn(TASK_DIM))
    a2 = hyper(torch.randn(TASK_DIM))
    k = next(iter(specs))
    dw1 = delta_w(*a1[k], hyper.scaling)
    dw2 = delta_w(*a2[k], hyper.scaling)
    assert not torch.allclose(dw1, dw2, atol=1e-4)


def test_reconstruction_overfit_decreases():
    """Full generate -> recon-loss -> backprop -> update loop reduces the loss."""
    _, specs, hyper = _setup()
    torch.manual_seed(1)
    target = {k: (torch.randn(4, specs[k][0]), torch.randn(specs[k][1], 4)) for k in specs}
    task_emb = torch.randn(TASK_DIM)
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
    cond_dim, in_f, out_f, r = 32, 48, 40, 8
    head = HEADS[name](cond_dim, in_f, out_f, r)
    a, b = head(torch.randn(cond_dim))
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
    cond_dim, r = 64, 16
    vera = estimate_params("vera", specs, cond_dim, r)
    lowrank = estimate_params("lowrank", specs, cond_dim, r)
    full = estimate_params("full", specs, cond_dim, r)
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
    res = idx.query("sentiment polarity", enc, top_k=3)
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)


# ---- Sprint 2: real generator (HyperLoRAGenerator) ------------------------
def _gen_setup(param="vera", d=24, n=2):
    from lora_lab.hypernet.model import HyperLoRAGenerator
    model = _TinyModel(d, n).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    specs = target_specs(model, TARGETS)
    gen = HyperLoRAGenerator(specs, task_dim=TASK_DIM, rank=4, alpha=8, parameterization=param,
                             trunk_hidden=32)
    return model, specs, gen


@pytest.mark.parametrize("param", ["vera", "lowrank", "full"])
def test_generator_shapes_and_noop(param):
    model, specs, gen = _gen_setup(param)
    adapter = gen(torch.randn(TASK_DIM))
    assert set(adapter) == set(specs)
    for key, (a, b) in adapter.items():
        in_f, out_f = specs[key]
        assert a.shape == (4, in_f) and b.shape == (out_f, 4)
        assert torch.allclose(b @ a, torch.zeros(out_f, in_f), atol=1e-6)  # no-op at init


def test_generator_conditions_on_task():
    """Different task embeddings -> different generated factors (A path is live)."""
    _, specs, gen = _gen_setup("vera")
    a1 = gen(torch.randn(TASK_DIM))
    a2 = gen(torch.randn(TASK_DIM))
    k = next(iter(specs))
    assert not torch.allclose(a1[k][0], a2[k][0], atol=1e-5)


def test_generator_apply_and_backprop():
    """Generated adapter injects onto the base and grads reach the generator."""
    model, _, gen = _gen_setup("vera")
    x = torch.randn(2, 4, 24)
    reg = LoRARegistry()
    reg.set_adapter(gen(torch.randn(TASK_DIM)))
    with lora_injected(model, TARGETS, reg, scaling=gen.scaling):
        out = model(x)
    out.pow(2).mean().backward()
    grads = [p.grad for p in gen.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_generator_layer_module_parsing():
    """Keys with 'layers.N' get distinct layer embeddings; modules distinct too."""
    _, _, gen = _gen_setup("vera", n=3)
    assert gen.layer_emb.num_embeddings == 3   # 3 layers in the tiny model
    assert set(gen.module_ids) == {"q_proj", "v_proj"}


# ---- Sprint 2/4: HyperConfig + the staged run YAMLs -----------------------
def test_hyperconfig_round_trip(tmp_path):
    from lora_lab.hypernet.config import HyperConfig
    c = HyperConfig(parameterization="vera", objective="sft", max_steps=10)
    p = c.save(tmp_path / "h.yaml")
    assert HyperConfig.load(p).to_dict() == c.to_dict()


def test_hyperconfig_rejects_bad_values():
    from lora_lab.hypernet.config import HyperConfig
    with pytest.raises(ValueError):
        HyperConfig(parameterization="bogus")
    with pytest.raises(ValueError):
        HyperConfig(objective="nope")


@pytest.mark.parametrize("name", ["tiny-plumbing", "recon-warmup", "sft-mistral"])
def test_staged_configs_load(name):
    """The committed configs/phase2/*.yaml stubs are valid HyperConfigs."""
    import pathlib
    from lora_lab.hypernet.config import HyperConfig
    path = pathlib.Path("configs/phase2") / f"{name}.yaml"
    if not path.exists():
        pytest.skip(f"{path} not present")
    cfg = HyperConfig.load(path)
    assert cfg.parameterization in ("vera", "lowrank", "full")
    assert cfg.objective in ("reconstruction", "sft")


# ---- Sprint 3/4: meta-training loop ---------------------------------------
class _TinyLM(nn.Module):
    """Minimal causal-LM-ish model with q_proj/v_proj targets + a .loss output."""
    def __init__(self, d=24, vocab=32):
        super().__init__()
        from torch.nn import functional as F  # noqa
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        self.attn = _TinyAttn(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        import types
        from torch.nn import functional as F
        h = self.embed(input_ids)
        h = h + self.attn(h)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab), labels.view(-1), ignore_index=-100)
        return types.SimpleNamespace(loss=loss, logits=logits)


def _fake_gen(model, task_dim=_FakeEncoder.dim):
    from lora_lab.hypernet.model import HyperLoRAGenerator
    specs = target_specs(model, TARGETS)
    return HyperLoRAGenerator(specs, task_dim=task_dim, rank=4, alpha=8,
                              parameterization="vera", trunk_hidden=16)


def test_meta_train_reconstruction_runs():
    from lora_lab.hypernet.meta_train import meta_train, SyntheticReconSampler
    model, specs, _ = _gen_setup("vera")
    gen = _fake_gen(model)
    sampler = SyntheticReconSampler(specs, rank=4, seed=0)
    losses = meta_train(gen, model, TARGETS, sampler, _FakeEncoder(),
                        steps=5, lr=1e-3, objective="reconstruction", log=lambda *_: None)
    assert len(losses) == 5 and all(torch.isfinite(torch.tensor(l)) for l in losses)


def test_meta_train_sft_runs_and_trains_generator():
    from lora_lab.hypernet.meta_train import meta_train
    torch.manual_seed(0)
    lm = _TinyLM(d=24)
    for p in lm.parameters():
        p.requires_grad_(False)
    gen = _fake_gen(lm)
    ids = torch.randint(0, 32, (2, 6))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

    class _FixedSFT:
        def sample(self):
            return "classify sentiment", batch

    before = [p.detach().clone() for p in gen.parameters()]
    losses = meta_train(gen, lm, TARGETS, _FixedSFT(), _FakeEncoder(),
                        steps=4, lr=1e-2, objective="sft", log=lambda *_: None)
    assert len(losses) == 4 and all(torch.isfinite(torch.tensor(l)) for l in losses)
    # generator actually updated (grads flowed through the frozen LM)
    after = list(gen.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
    # base stayed frozen
    assert all(p.grad is None for p in lm.parameters())


def test_reconstruction_minibatch_averages_over_dataset_and_learns():
    """Standard SGD: each step averages recon error over a minibatch of LoRAs
    drawn from the training set, and that averaged loss drives the weights down.

    Guards the fix for the 1-arbitrary-LoRA-per-step bug (loss pinned at 1.0)."""
    from lora_lab.hypernet.meta_train import meta_train
    from lora_lab.hypernet.model import HyperLoRAGenerator
    torch.manual_seed(0)
    model = _TinyModel(24, 2).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    specs = target_specs(model, TARGETS)
    gen = HyperLoRAGenerator(specs, task_dim=_FakeEncoder.dim, rank=4, alpha=8,
                             parameterization="lowrank", trunk_hidden=16)

    # a tiny fixed 'database' of 3 LoRAs, each tied to a distinct description
    pool = []
    for desc in ("classify sentiment", "detect entailment", "do translation"):
        target = {k: (torch.randn(4, specs[k][0]) * 0.1, torch.randn(specs[k][1], 4) * 0.1)
                  for k in specs}
        pool.append((desc, target))

    class _CountingFixedSampler:
        def __init__(self):
            self.calls = 0
        def sample(self):
            item = pool[self.calls % len(pool)]
            self.calls += 1
            return item

    sampler = _CountingFixedSampler()
    losses = meta_train(gen, model, TARGETS, sampler, _FakeEncoder(),
                        steps=40, lr=1e-2, objective="reconstruction", batch_size=3,
                        log=lambda *_: None)
    assert len(losses) == 40
    # batch_size LoRAs are drawn every step (the minibatch), not one
    assert sampler.calls == 40 * 3
    # the averaged loss over the dataset drops well off the 1.0 no-learning floor
    assert losses[0] > 0.9 and losses[-1] < 0.6 * losses[0]


# ---- Sprint 3/4: real samplers (parse + leakage filter, no network) -------
def test_parse_lora_state_dict_maps_to_base_keys():
    from lora_lab.hypernet.samplers import parse_lora_state_dict
    r, in_f, out_f = 16, 4096, 1024
    base = "model.layers.3.self_attn.v_proj"
    sd = {
        f"base_model.model.{base}.lora_A.weight": torch.randn(r, in_f),
        f"base_model.model.{base}.lora_B.weight": torch.randn(out_f, r),
        # an unrelated key should be ignored
        "base_model.model.model.embed_tokens.weight": torch.randn(8, 8),
    }
    parsed = parse_lora_state_dict(sd)
    assert set(parsed) == {base}
    a, b = parsed[base]
    assert a.shape == (r, in_f) and b.shape == (out_f, r)


def test_recon_sampler_train_only(tmp_path):
    """Sampler covers only train-split tasks (held-out never enters)."""
    import yaml
    from lora_lab.hypernet.samplers import LibraryReconSampler
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({
        "train": {"taskA": "taskA_x", "taskB": "taskB_y"},
        "held_out": {"taskC": "taskC_z"},
    }))
    lib = tmp_path / "lib.yaml"
    lib.write_text(yaml.safe_dump({"tasks": {
        "taskA": {"adapter_repo": "R/A", "description": "do A"},
        "taskB": {"adapter_repo": "R/B", "description": "do B"},
        "taskC": {"adapter_repo": "R/C", "description": "do C"},  # held-out
    }}))
    s = LibraryReconSampler(split, lib, seed=0)
    assert set(s.tasks) == {"taskA", "taskB"}        # train only
    assert "taskC" not in s.tasks                     # held-out excluded
    assert len(s) == 2


# ---- Sprint 4: entrypoint GPU guard ---------------------------------------
def test_run_guard_blocks_gpu_without_flag():
    from lora_lab.hypernet.config import HyperConfig
    from lora_lab.hypernet.meta_train import assert_run_allowed
    gpu_cfg = HyperConfig(device="cuda", load_in_4bit=True)
    with pytest.raises(RuntimeError):
        assert_run_allowed(gpu_cfg, allow_gpu=False)   # refuses autonomous GPU run
    assert_run_allowed(gpu_cfg, allow_gpu=True)         # ok when user opts in
    cpu_cfg = HyperConfig(device="cpu", load_in_4bit=False)
    assert_run_allowed(cpu_cfg, allow_gpu=False)        # cpu always allowed


# ---- Sprint 1: W&B / RunLogger contract -----------------------------------
def _hyper_cfg(tmp_path, objective, **kw):
    from lora_lab.hypernet.config import HyperConfig
    return HyperConfig(objective=objective, device="cpu", load_in_4bit=False,
                       wandb_mode="disabled", output_root=str(tmp_path),
                       run_name=f"t-{objective}", base_model="tiny/stub", **kw)


def test_meta_train_logs_metrics_recon(tmp_path):
    import json
    from lora_lab.hypernet.logging import build_run_logger
    from lora_lab.hypernet.meta_train import meta_train, SyntheticReconSampler
    model, specs, _ = _gen_setup("vera")
    gen = _fake_gen(model)
    cfg = _hyper_cfg(tmp_path, "reconstruction")
    logger = build_run_logger(cfg, stage="S1")
    meta_train(gen, model, TARGETS, SyntheticReconSampler(specs, rank=4), _FakeEncoder(),
               steps=4, lr=1e-3, objective="reconstruction", logger=logger, log=lambda *_: None)
    logger.finish()
    recs = [json.loads(l) for l in (logger.output_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert len(recs) == 4
    assert all("reconstruction/loss" in r and "gpu_mem_gb" in r for r in recs)
    assert (logger.output_dir / "config.yaml").exists()
    assert (logger.output_dir / "summary.json").exists()  # offline-fallback still writes locally


def test_meta_train_logs_metrics_sft(tmp_path):
    import json
    from lora_lab.hypernet.logging import build_run_logger
    from lora_lab.hypernet.meta_train import meta_train
    lm = _TinyLM(d=24)
    for p in lm.parameters():
        p.requires_grad_(False)
    gen = _fake_gen(lm)
    ids = torch.randint(0, 32, (2, 6))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

    class _FixedSFT:
        def sample(self):
            return "classify sentiment", batch

    cfg = _hyper_cfg(tmp_path, "sft")
    logger = build_run_logger(cfg, stage="S1")
    meta_train(gen, lm, TARGETS, _FixedSFT(), _FakeEncoder(),
               steps=3, lr=1e-2, objective="sft", logger=logger, log=lambda *_: None)
    logger.finish()
    recs = [json.loads(l) for l in (logger.output_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert len(recs) == 3 and all("sft/loss" in r and "grad_norm" in r for r in recs)


# ---- Sprint 2: real-spec size report (T3 inputs) --------------------------
def test_mistral_spec_size_report():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("scripts")))
    from phase2_size_report import mistral_target_specs, size_rows
    specs = mistral_target_specs()
    assert len(specs) == 96  # q/k/v x 32 layers
    assert specs["model.layers.0.self_attn.k_proj"] == (4096, 1024)  # GQA
    rows = {r["parameterization"]: r["hypernet_params_M"] for r in size_rows()}
    assert rows["vera"] < rows["lowrank"] < rows["full"]  # the S2 ladder ordering
