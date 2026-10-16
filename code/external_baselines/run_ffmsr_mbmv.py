"""Run the authors' FFMSR implementation on Music, Books, and Movies.

The model and optimization code are imported unchanged from the official
repository.  This wrapper only supplies the FedPASA data paths, deterministic
seeds, GPU selection, full-ranking cutoffs, and a validation/test split that
does not use test labels for early stopping.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import torch

# RecBole 1.2 predates PyTorch 2.6, whose default checkpoint loader changed
# to weights_only=True.  FFMSR checkpoints contain a RecBole Config object,
# so preserve the legacy behavior for checkpoints produced locally here.
_torch_load = torch.load


def _recbole_compatible_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _recbole_compatible_load


def ensure_recbole_layout() -> None:
    """Materialize the directory layout expected by RecBole without duplicating features."""
    for domain in ("Music", "Books", "Movies"):
        domain_dir = DATA / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("train.inter", "valid.inter", "test.inter", "text", "item2index"):
            source = DATA / f"{domain}.{suffix}"
            destination = domain_dir / source.name
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                shutil.copy2(source, destination)
        source_feat = DATA / f"{domain}.feat1CLS"
        destination_feat = domain_dir / source_feat.name
        if not destination_feat.exists():
            os.link(source_feat, destination_feat)


HERE = Path(__file__).resolve().parent
REPO = HERE / "FFMSR"
DATA = HERE / "recbole_mbmv"
OUTPUT = HERE / "results" / "ffmsr"
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from recbole.config import Config  # noqa: E402
from recbole.data import data_preparation  # noqa: E402
from recbole.utils import init_logger, init_seed  # noqa: E402
from Trainer.kmeansplus_trainer import FedtrainTrainer  # noqa: E402
from Trainer.single_trainer import SingleTrainer  # noqa: E402
from data.dataset import FederatedDataset  # noqa: E402
from model.FFMSR_fed import MLTRec  # noqa: E402


def common(seed: int, result_dir: Path) -> dict:
    return {
        "data_path": str(DATA).replace("\\", "/") + "/",
        "checkpoint_dir": str(result_dir).replace("\\", "/"),
        "gpu_id": 0,
        "use_gpu": True,
        "device": "cuda:0",
        "seed": seed,
        "reproducibility": True,
        "topk": [10, 20],
        "metrics": ["Recall", "NDCG"],
        "valid_metric": "Recall@10",
        "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "mode": "full"},
        "train_neg_sample_args": None,
        "show_progress": True,
    }


def make_config(domain: str, stage: str, seed: int, result_dir: Path,
                smoke: bool, pretrain_epochs: int, finetune_epochs: int,
                books_batch_size: int, books_ft_layer: int) -> Config:
    props = [str(REPO / "props" / "FFMSR.yaml"), str(REPO / "props" / f"{stage}.yaml")]
    overrides = common(seed, result_dir)
    if stage == "pretrain":
        overrides.update(
            {
                "pretrain_epochs": 1 if smoke else pretrain_epochs,
                "save_step": 1 if smoke else pretrain_epochs,
                "cluster_centroids": 8 if smoke else 120,
                "cluster_iters": 2 if smoke else 300,
            }
        )
    else:
        overrides.update({"epochs": 1 if smoke else finetune_epochs,
                          "stopping_step": 2 if smoke else 10})
    if domain == "Books":
        overrides["train_batch_size"] = books_batch_size
        overrides["ft_layer"] = books_ft_layer
    return Config(model=MLTRec, dataset=domain, config_file_list=props, config_dict=overrides)


def build(domain: str, stage: str, seed: int, result_dir: Path,
          smoke: bool, pretrain_epochs: int, finetune_epochs: int,
          books_batch_size: int, books_ft_layer: int):
    config = make_config(domain, stage, seed, result_dir, smoke, pretrain_epochs,
                         finetune_epochs, books_batch_size, books_ft_layer)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = FederatedDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = MLTRec(config, train_data.dataset).to(config["device"])
    # The model owns a deep copy. Release the dataset's duplicate CUDA feature
    # tensor; loaders only use IDs and interaction sequences after construction.
    dataset.layer_embedding = torch.empty(0)
    return config, train_data, valid_data, test_data, model


def run_pair(source: str, target: str, seed: int, smoke: bool,
             pretrain_epochs: int, finetune_epochs: int,
             books_batch_size: int, books_ft_layer: int) -> dict:
    pair_dir = OUTPUT / f"{source.lower()}_{target.lower()}" / f"seed{seed}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    config_s, train_s, _, _, model_s = build(
        source, "pretrain", seed, pair_dir, smoke, pretrain_epochs,
        finetune_epochs, books_batch_size, books_ft_layer)
    config_t, train_t, _, _, model_t = build(
        target, "pretrain", seed, pair_dir, smoke, pretrain_epochs,
        finetune_epochs, books_batch_size, books_ft_layer)
    trainer = FedtrainTrainer(config_s, config_t, model_s, model_t)
    trainer.fedtrain(train_s, train_t, show_progress=True)

    epoch = 1 if smoke else pretrain_epochs
    stamp = str(date.today())
    checkpoints = {
        source: pair_dir / f"MLTRec-{source}-{epoch}-{stamp}.pth",
        target: pair_dir / f"MLTRec-{target}-{epoch}-{stamp}.pth",
    }
    results = {}
    del trainer, model_s, model_t, train_s, train_t
    torch.cuda.empty_cache()

    for domain in (source, target):
        config, train_data, valid_data, test_data, model = build(
            domain, "finetune", seed, pair_dir, smoke, pretrain_epochs,
            finetune_epochs, books_batch_size, books_ft_layer)
        checkpoint = torch.load(checkpoints[domain], map_location=config["device"], weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        fine_trainer = SingleTrainer(config, model)
        best_valid_score, best_valid_result = fine_trainer.fit(
            train_data, valid_data, saved=True, show_progress=config["show_progress"]
        )
        test_result = fine_trainer.evaluate(
            test_data, load_best_model=True, show_progress=config["show_progress"]
        )
        results[domain] = {
            "best_valid_score": float(best_valid_score),
            "best_valid_result": {k: float(v) for k, v in best_valid_result.items()},
            "test_result": {k: float(v) for k, v in test_result.items()},
            "checkpoint": str(checkpoints[domain]),
        }
        del fine_trainer, model, train_data, valid_data, test_data
        torch.cuda.empty_cache()

    payload = {
        "method": "FFMSR",
        "official_commit": "e5b6d59f1abb73e31d051d03ef9fc82763400f2e",
        "domains": [source, target],
        "seed": seed,
        "subset_users_per_domain": 5000,
        "subset_selection_seed": 777,
        "evaluation": "full ranking",
        "precision": "CUDA bfloat16 automatic mixed precision; FP32 optimizer states",
        "pretrain_epochs": 1 if smoke else pretrain_epochs,
        "finetune_epochs": 1 if smoke else finetune_epochs,
        "books_train_batch_size": books_batch_size,
        "books_bert_layers": books_ft_layer,
        "smoke": smoke,
        "results": results,
    }
    (pair_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["Music", "Books", "Movies"], required=True)
    parser.add_argument("--target", choices=["Music", "Books", "Movies"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20,
                        help="Maximum fine-tuning epochs; validation selects the checkpoint.")
    parser.add_argument("--books-batch-size", type=int, default=2048,
                        help="Books batch size; the objective and full catalog are unchanged.")
    parser.add_argument("--books-ft-layer", type=int, default=1,
                        help="Number of final BERT layers used for Books on 16 GB GPUs.")
    args = parser.parse_args()
    if args.source == args.target:
        parser.error("source and target must differ")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by the experiment policy")
    ensure_recbole_layout()
    print(json.dumps(run_pair(args.source, args.target, args.seed, args.smoke,
                              args.pretrain_epochs, args.finetune_epochs,
                              args.books_batch_size, args.books_ft_layer), indent=2))


if __name__ == "__main__":
    main()
