"""Re-evaluate fine-tuned FFMSR checkpoints under RecBole's uni99 sampled protocol.

For every pair/seed directory produced by ``run_ffmsr_mbmv.py`` this script
loads the best-validation fine-tuned checkpoint of each domain and evaluates
the test set with ``eval_args.mode = uni99`` (1 positive + 99 uniformly
sampled negatives, negatives drawn by RecBole's sampler which excludes the
user's observed items; RNG seeded with the run seed).

Wrapper-only patch (official sources untouched): RecBole sampled evaluation
calls ``model.predict``, which MLTRec inherits from SASRec — that method both
has an incompatible signature and would score with the raw ID embedding only.
We bind a ``predict`` that gathers candidate scores from the model's own
``full_sort_predict``, so the uni99 ranking uses exactly the same scoring
function as the full-ranking evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

_torch_load = torch.load


def _recbole_compatible_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _recbole_compatible_load

HERE = Path(__file__).resolve().parent
REPO = HERE / "FFMSR"
DATA = HERE / "recbole_mbmv"
PAIRS_ROOT = HERE / "results" / "ffmsr"
OUT_ROOT = HERE.parent / "outputs" / "external_baselines" / "ffmsr"
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from recbole.config import Config  # noqa: E402
from recbole.data import data_preparation  # noqa: E402
from recbole.utils import init_seed  # noqa: E402
from Trainer.single_trainer import SingleTrainer  # noqa: E402
from data.dataset import FederatedDataset  # noqa: E402
from model.FFMSR_fed import MLTRec  # noqa: E402

NAME = {"Music": "music", "Books": "books", "Movies": "movies"}
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def is_finetuned_checkpoint(path: Path) -> bool:
    parts = path.stem.split("-")
    return len(parts) > 1 and parts[1] in MONTHS


def find_finetuned_checkpoints(pair_dir: Path) -> dict:
    """Map domain -> latest fine-tuned (best-valid) checkpoint in a pair dir."""
    by_domain: dict[str, list[Path]] = {}
    for path in sorted(pair_dir.glob("MLTRec-*.pth")):
        if not is_finetuned_checkpoint(path):
            continue
        cfg = torch.load(path, map_location="cpu")["config"]
        by_domain.setdefault(cfg["dataset"], []).append(path)
    return {d: max(paths, key=lambda p: p.stat().st_mtime) for d, paths in by_domain.items()}


def predict_via_full_sort(self, interaction):
    scores = self.full_sort_predict(interaction)
    test_item = interaction[self.ITEM_ID]
    return scores.gather(1, test_item.view(-1, 1)).squeeze(1)


def build_config(domain: str, seed: int, mode: str, books_ft_layer: int) -> Config:
    props = [str(REPO / "props" / "FFMSR.yaml"), str(REPO / "props" / "finetune.yaml")]
    overrides = {
        "data_path": str(DATA).replace("\\", "/") + "/",
        "checkpoint_dir": str(HERE / "saved_uni99").replace("\\", "/"),
        "gpu_id": 0,
        "use_gpu": True,
        "device": "cuda:0",
        "seed": seed,
        "reproducibility": True,
        "topk": [10, 20],
        "metrics": ["Recall", "NDCG"],
        "valid_metric": "Recall@10",
        "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "mode": mode},
        "train_neg_sample_args": None,
        "show_progress": True,
    }
    if domain == "Books":
        overrides["ft_layer"] = books_ft_layer
    return Config(model=MLTRec, dataset=domain, config_file_list=props, config_dict=overrides)


def evaluate_domain(domain: str, checkpoint: Path, seed: int, mode: str,
                    books_ft_layer: int) -> dict:
    config = build_config(domain, seed, mode, books_ft_layer)
    init_seed(config["seed"], config["reproducibility"])
    dataset = FederatedDataset(config)
    _, _, test_data = data_preparation(config, dataset)
    model = MLTRec(config, test_data.dataset).to(config["device"])
    model.predict = predict_via_full_sort.__get__(model, MLTRec)
    state = torch.load(checkpoint, map_location=config["device"])
    model.load_state_dict(state["state_dict"])
    dataset.layer_embedding = torch.empty(0)
    trainer = SingleTrainer(config, model)
    result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    return {k: float(v) for k, v in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="uni99")
    parser.add_argument("--pairs", nargs="*",
                        default=["music_movies", "music_books", "books_movies"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123])
    args = parser.parse_args()

    for pair in args.pairs:
        for seed in args.seeds:
            pair_dir = PAIRS_ROOT / pair / f"seed{seed}"
            payload = json.loads((pair_dir / "results.json").read_text(encoding="utf-8"))
            books_ft_layer = int(payload.get("books_bert_layers") or 3)
            checkpoints = find_finetuned_checkpoints(pair_dir)
            print(f"[{pair} seed{seed}] checkpoints: "
                  f"{ {d: p.name for d, p in checkpoints.items()} }", flush=True)
            domains = pair.split("_")
            for target_dir in domains:
                domain = next(d for d in ("Music", "Books", "Movies")
                              if NAME[d] == target_dir)
                source_dir = next(d for d in domains if d != target_dir)
                direction = f"{source_dir}2{target_dir}"
                out_file = OUT_ROOT / direction / f"seed{seed}" / f"result_{args.mode}.json"
                if out_file.exists():
                    print(f"  skip {direction} seed{seed} (exists)", flush=True)
                    continue
                result = evaluate_domain(domain, checkpoints[domain], seed,
                                         args.mode, books_ft_layer)
                record = {
                    "method": "FFMSR (TOIS 2025, official code e5b6d59)",
                    "direction": direction,
                    "source_domain": next(d for d in ("Music", "Books", "Movies")
                                          if NAME[d] == source_dir),
                    "target_domain": domain,
                    "seed": seed,
                    "evaluation": f"RecBole sampled evaluation, mode={args.mode} "
                                  "(1 positive + 99 uniform negatives, "
                                  "observed items excluded by RecBole sampler, "
                                  "sampler RNG seeded with the run seed)",
                    "scoring": "candidate scores gathered from the model's own "
                               "full_sort_predict (wrapper-only predict binding; "
                               "identical scoring function as full-ranking eval)",
                    "checkpoint": str(checkpoints[domain]),
                    "metrics": {
                        "test": {
                            "HR@10": result["recall@10"],
                            "HR@20": result["recall@20"],
                            "NDCG@10": result["ndcg@10"],
                            "NDCG@20": result["ndcg@20"],
                        }
                    },
                    "raw": result,
                }
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
                print(f"  {direction} seed{seed}: {result}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
