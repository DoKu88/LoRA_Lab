#!/usr/bin/env python
"""Evaluate a trained Text-to-LoRA hypernetwork.

For a chosen task, three models are compared on the *same* input, all applied
live to the same frozen base:

    base       the frozen base model, no LoRA
    oracle     base + the real library LoRA we are trying to replicate
    generated  base + the LoRA the hypernetwork generates from the task description

Inputs:
    --config   a training run's config.yaml (architecture + encoder + data paths).
               The checkpoint is the newest *.pt next to it (override with --checkpoint).
    --tests    a small YAML test spec (which tasks / how many examples) -> batch mode.
    --interactive   pick a task, type prompts, see all three outputs live.

Examples:
    python scripts/eval_hypernet.py --config results/phase2/runs/<run>/config.yaml --interactive
    python scripts/eval_hypernet.py --config results/.../config.yaml --tests configs/phase2/eval-tests.yaml
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lora_lab.data.task_dataset import TaskSpec, get_dataset
from lora_lab.eval.evaluate import _generate_and_score
from lora_lab.hypernet.apply import LoRARegistry, inject, target_specs
from lora_lab.hypernet.config import HyperConfig
from lora_lab.hypernet.data import _load_library, _split_tasks, parse_lora_state_dict
from lora_lab.hypernet.model import HyperLoRAGenerator, MeanPoolEncoder

# ---- terminal styling (no dependency) -------------------------------------
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
C = {"base": "\033[90m", "oracle": "\033[32m", "generated": "\033[36m",
     "accent": "\033[35m", "title": "\033[1;37m", "warn": "\033[33m"}
LABEL = {"base": "BASE  (no LoRA)", "oracle": "ORACLE  (the LoRA we replicate)",
         "generated": "GENERATED  (hypernetwork)"}
WIDTH = 88
WRAP = 120   # wrap width for printed text; override with --width


def rule(text: str = "", color: str = "accent") -> None:
    c = C[color]
    if text:
        print(f"{c}{'─' * 2} {text} {'─' * max(0, WIDTH - len(text) - 4)}{RESET}")
    else:
        print(f"{c}{'─' * WIDTH}{RESET}")


def block(label: str, text: str, color: str) -> None:
    print(f"{C[color]}{BOLD}▌ {label}{RESET}")
    for line in textwrap.wrap(text, WRAP) or [f"{DIM}(empty){RESET}"]:
        print(f"    {line}")
    print()


def labeled(label: str, text: str, color: str, lw: int = 10) -> None:
    """Print 'label  text' with the full text wrapped at WRAP; continuation lines
    align under the text."""
    lines = textwrap.wrap(str(text), WRAP) or [""]
    print(f"  {color}{label:<{lw}}{RESET} {lines[0]}")
    for line in lines[1:]:
        print(f"  {'':<{lw}} {line}")


def legend(include_gold: bool = True) -> None:
    """Describe each output line once, up front."""
    rows = [("input", "accent", "the prompt sent to every model (task instructions + few-shot + the question)")]
    if include_gold:
        rows.append(("gold", "accent", "the dataset's correct answer — what the models are scored against"))
    rows += [
        ("base", "base", "the bare base model — NO LoRA"),
        ("oracle", "oracle", "base + the real library LoRA we are replicating (the target behaviour)"),
        ("generated", "generated", "base + the LoRA the hypernetwork generated from the description"),
    ]
    print(f"  {BOLD}legend{RESET}")
    for label, color, desc in rows:
        print(f"    {C[color]}{label:<10}{RESET} {DIM}{desc}{RESET}")
    print()


# ---- the evaluator --------------------------------------------------------
class HypernetEvaluator:
    def __init__(self, config_path: str, checkpoint_path: str | None):
        cfg_path, self.ckpt = _resolve_paths(config_path, checkpoint_path)
        self.cfg = HyperConfig.load(cfg_path)
        self.device = self.cfg.device
        self.library = _load_library(self.cfg.library_path)

        print(f"{DIM}loading base {self.cfg.base_model} (4bit={self.cfg.load_in_4bit}) ...{RESET}")
        self.tok = AutoTokenizer.from_pretrained(self.cfg.base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if self.cfg.load_in_4bit:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16)
            self.base = AutoModelForCausalLM.from_pretrained(
                self.cfg.base_model, quantization_config=bnb, device_map={"": 0})
        else:
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            self.base = AutoModelForCausalLM.from_pretrained(self.cfg.base_model, dtype=dtype).to(self.device)
        self.base.eval()

        print(f"{DIM}loading encoder {self.cfg.encoder_model} + hypernet {self.ckpt.name} ...{RESET}")
        self.encoder = MeanPoolEncoder(self.cfg.encoder_model, device=self.device)
        specs = target_specs(self.base, self.cfg.target_modules)
        self.generator = HyperLoRAGenerator(
            specs, task_dim=self.encoder.dim, rank=self.cfg.rank, alpha=self.cfg.alpha,
            parameterization=self.cfg.parameterization, layer_dim=self.cfg.layer_dim,
            module_dim=self.cfg.module_dim, trunk_hidden=self.cfg.trunk_hidden).to(self.device).eval()
        self.generator.load_state_dict(torch.load(self.ckpt, map_location=self.device))

        # inject once with scaling=1; each adapter folds its own scaling into B.
        self.registry = LoRARegistry()
        inject(self.base, self.cfg.target_modules, self.registry, scaling=1.0)
        self._oracle_cache: dict[str, dict] = {}

    # -- adapters: {key: (A, scaling*B)} so the registry (scaling 1) is faithful --
    @torch.no_grad()
    def generate_adapter(self, description: str) -> dict:
        emb = self.encoder.encode([description]).to(self.device).squeeze(0)
        s = self.generator.scaling
        return {k: (a.detach(), (s * b).detach()) for k, (a, b) in self.generator(emb).items()}

    def oracle_adapter(self, task: str) -> dict | None:
        repo = self.library.get(task, {}).get("adapter_repo")
        if not repo:
            return None
        if task not in self._oracle_cache:
            raw = parse_lora_state_dict(load_file(hf_hub_download(repo, "adapter_model.safetensors")))
            acfg = json.load(open(hf_hub_download(repo, "adapter_config.json")))
            s = acfg.get("lora_alpha", acfg.get("r", 16)) / acfg.get("r", 16)  # the oracle's own scaling
            self._oracle_cache[task] = {k: (a.to(self.device), (s * b).to(self.device))
                                        for k, (a, b) in raw.items()}
        return self._oracle_cache[task]

    def _select(self, which: str, gen: dict, oracle: dict | None) -> None:
        if which == "base":
            self.registry.clear()
        elif which == "generated":
            self.registry.set_adapter(gen)
        elif which == "oracle":
            self.registry.set_adapter(oracle)

    @torch.no_grad()
    def generate_text(self, which: str, prompt: str, gen: dict, oracle: dict | None,
                      max_new_tokens: int) -> str:
        self._select(which, gen, oracle)
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):       # some versions return a dict-like
            ids = ids["input_ids"]
        ids = ids.to(self.device)
        out = self.base.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def _latest_pt(run_dir: Path) -> Path | None:
    pts = sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return pts[-1] if pts else None


def _resolve_paths(config_arg: str, checkpoint_arg: str | None) -> tuple[Path, Path]:
    """Accept --config as a run dir, a config.yaml, or a checkpoint .pt.

    Returns (config.yaml, checkpoint). The config.yaml is read from the run
    directory; the checkpoint is --checkpoint, or a .pt passed as --config, else
    the newest *.pt in that directory.
    """
    p = Path(config_arg)
    if p.is_dir():
        cfg, ckpt = p / "config.yaml", (Path(checkpoint_arg) if checkpoint_arg else _latest_pt(p))
    elif p.suffix == ".pt":
        ckpt, cfg = (Path(checkpoint_arg) if checkpoint_arg else p), p.parent / "config.yaml"
    else:  # a config.yaml
        cfg, ckpt = p, (Path(checkpoint_arg) if checkpoint_arg else _latest_pt(p.parent))
    if not cfg.exists():
        raise SystemExit(f"config.yaml not found at {cfg}")
    if not ckpt or not Path(ckpt).exists():
        raise SystemExit(f"no checkpoint (.pt) found for --config {config_arg}")
    return cfg, Path(ckpt)


# ---- interactive mode -----------------------------------------------------
def _default_task(ev: HypernetEvaluator) -> str:
    for t in _split_tasks(ev.cfg.split_path, "train"):
        s = ev.library.get(t, {})
        if s.get("description") and s.get("adapter_repo"):
            return t
    raise SystemExit("no usable train task found in the library")


def interactive(ev: HypernetEvaluator, task: str | None, max_new_tokens: int) -> None:
    task = task or _default_task(ev)

    def prepare(task_name: str, custom_desc: str | None = None):
        spec = ev.library.get(task_name, {})
        desc = custom_desc or spec.get("description", "")
        gen = ev.generate_adapter(desc)
        oracle = None if custom_desc else ev.oracle_adapter(task_name)
        return desc, gen, oracle

    desc, gen, oracle = prepare(task)
    rule("interactive eval", "title")
    print(f"  {C['accent']}task{RESET}        {task}")
    print(f"  {C['accent']}description{RESET} {desc}")
    print(f"  {C['warn']}commands{RESET}    :task <name>   :desc <text>   :tokens <n>   :help   :quit\n")
    print(f"  type an {BOLD}input{RESET} and press enter to compare base / oracle / generated.\n")
    legend(include_gold=False)

    while True:
        try:
            line = input(f"{C['title']}input› {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q"):
            break
        if line in (":help", ":h"):
            print("  :task <name>   switch task (loads its description + oracle)\n"
                  "  :desc <text>   use a free-text description (no oracle)\n"
                  "  :tokens <n>    set max new tokens\n  :quit\n")
            continue
        if line.startswith(":tokens "):
            max_new_tokens = int(line.split(maxsplit=1)[1]); print(f"  max_new_tokens = {max_new_tokens}\n"); continue
        if line.startswith(":task "):
            task = line.split(maxsplit=1)[1].strip()
            if task not in ev.library:
                print(f"  {C['warn']}unknown task {task!r}{RESET}\n"); continue
            desc, gen, oracle = prepare(task)
            print(f"  {C['accent']}task{RESET} {task}\n  {C['accent']}description{RESET} {desc}\n"); continue
        if line.startswith(":desc "):
            desc, gen, oracle = prepare(task, custom_desc=line.split(maxsplit=1)[1].strip())
            print(f"  {C['accent']}custom description{RESET} {desc}  {DIM}(oracle unavailable){RESET}\n"); continue

        # otherwise: `line` is the input prompt
        print()
        rule(f"input: {line}", "accent")
        for which in ("base", "oracle", "generated"):
            if which == "oracle" and oracle is None:
                continue
            out = ev.generate_text(which, line, gen, oracle, max_new_tokens)
            block(LABEL[which], out, which)


# ---- batch mode -----------------------------------------------------------
def batch(ev: HypernetEvaluator, tests: dict, default_max_new_tokens: int) -> None:
    tasks = tests.get("tasks") or _split_tasks(ev.cfg.split_path, tests.get("split", "val"))
    n = int(tests.get("n_examples", 3))
    mnt = int(tests.get("max_new_tokens", default_max_new_tokens))
    models = tests.get("models", ["base", "oracle", "generated"])
    show_samples = tests.get("show_samples", True)

    rule("batch eval", "title")
    print(f"  {len(tasks)} task(s), {n} example(s) each, models={models}\n")
    legend(include_gold=True)
    rows = []
    for task in tasks:
        spec = ev.library.get(task)
        if not (spec and spec.get("dataset_repo") and spec.get("description")):
            print(f"  {C['warn']}skip {task} (no dataset/description){RESET}"); continue
        metric = spec.get("metric", "rougeL")
        desc = spec["description"]
        bundle = get_dataset(TaskSpec(name=task, hf_repo=spec["dataset_repo"],
                                      kind=spec.get("kind", "generation"), metric=metric, description=desc),
                             ev.tok, max_seq_len=ev.cfg.max_seq_len, max_eval_samples=n)
        examples = bundle.test_eval[:n]
        if not examples:
            print(f"  {C['warn']}skip {task} (no eval examples){RESET}"); continue
        gen = ev.generate_adapter(desc)
        oracle = ev.oracle_adapter(task)

        scores, sample = {}, None
        for which in models:
            if which == "oracle" and oracle is None:
                continue
            ev._select(which, gen, oracle)
            scored = _generate_and_score(ev.base, ev.tok, examples, metric,
                                         max_new_tokens=mnt, batch_size=min(8, len(examples)))
            scores[which] = scored["score"]
            if sample is None:
                sample = {"input": examples[0]["input_text"], "ref": examples[0]["references"][0]}
            sample[which] = scored["sample_predictions"][0]
        rows.append((task, metric, scores))

        rule(f"{task}   [{metric}]", "accent")
        if show_samples and sample:
            labeled("input", sample["input"], C["accent"])
            labeled("gold", sample["ref"], C["accent"])
            for which in models:
                if which in sample and which in scores:
                    labeled(which, sample[which], C[which])
        print("  " + "  ".join(f"{C[w]}{w}={scores[w]:.3f}{RESET}" for w in models if w in scores) + "\n")

    # summary
    rule("summary", "title")
    hdr = f"  {'task':<42} {'metric':<10} " + " ".join(f"{m:>10}" for m in models)
    print(f"{BOLD}{hdr}{RESET}")
    for task, metric, scores in rows:
        cells = " ".join(f"{scores.get(m, float('nan')):>10.3f}" for m in models)
        print(f"  {task:<42} {metric:<10} {cells}")
    if rows:
        print(f"  {DIM}{'mean':<42} {'':<10} " +
              " ".join(f"{sum(s.get(m, 0) for _, _, s in rows) / len(rows):>10.3f}" for m in models) + RESET)


def main() -> int:
    global WRAP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="a run's config.yaml, its run directory, or a checkpoint .pt")
    ap.add_argument("--checkpoint", default=None, help="hypernet .pt (default: newest in the run dir)")
    ap.add_argument("--tests", default=None, help="YAML test spec -> batch mode")
    ap.add_argument("--interactive", action="store_true", help="interactive prompt loop")
    ap.add_argument("--task", default=None, help="preselect a task (interactive)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--width", type=int, default=WRAP, help="wrap width for printed text (default 80)")
    args = ap.parse_args()
    WRAP = args.width

    ev = HypernetEvaluator(args.config, args.checkpoint)
    if args.tests:
        batch(ev, yaml.safe_load(Path(args.tests).read_text()) or {}, args.max_new_tokens)
    else:
        interactive(ev, args.task, args.max_new_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
