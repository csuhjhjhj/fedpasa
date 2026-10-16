"""Generate the exact 12-layer BERT CLS features expected by FFMSR.

This follows the authors' preprocessing routine: bert-base-uncased,
``output_hidden_states=True``, and concatenated CLS vectors from the 12
Transformer layers.  Output is a raw float32 file named ``*.feat1CLS``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_texts(path: Path) -> list[str]:
    rows: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            item, text = line.rstrip("\n").split("\t", 1)
            rows[int(item)] = text
    return [rows[index] for index in range(len(rows))]


def encode_domain(
    data_dir: Path,
    domain: str,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> None:
    text_path = data_dir / f"{domain}.text"
    output_path = data_dir / f"{domain}.feat1CLS"
    progress_path = data_dir / f"{domain}.feat1CLS.progress.json"
    texts = load_texts(text_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).cuda().eval()
    if model.config.hidden_size != 768 or model.config.num_hidden_layers != 12:
        raise RuntimeError("FFMSR requires a 12-layer, 768-dimensional BERT encoder")

    expected_bytes = len(texts) * 12 * 768 * np.dtype(np.float32).itemsize
    start = 0
    mode = "w+"
    if output_path.exists() and output_path.stat().st_size == expected_bytes and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start = int(progress.get("next_index", 0))
        mode = "r+"
    features = np.memmap(output_path, dtype=np.float32, mode=mode, shape=(len(texts), 12, 768))
    for begin in tqdm(range(start, len(texts), batch_size), desc=f"BERT features: {domain}"):
        batch = texts[begin: begin + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.cuda(non_blocking=True) for key, value in tokens.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**tokens, output_hidden_states=True)
            cls = torch.stack([layer[:, 0, :] for layer in outputs.hidden_states[1:]], dim=1)
        end = begin + len(batch)
        features[begin:end] = cls.float().cpu().numpy()
        features.flush()
        progress_path.write_text(
            json.dumps({"domain": domain, "next_index": end, "items": len(texts)}, indent=2),
            encoding="utf-8",
        )
    del features
    progress_path.write_text(
        json.dumps({"domain": domain, "next_index": len(texts), "items": len(texts), "complete": True}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "recbole_mbmv",
    )
    parser.add_argument("--domains", nargs="+", default=["Music", "Books", "Movies"])
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FFMSR feature extraction")
    for domain in args.domains:
        encode_domain(args.data_dir, domain, args.model, args.batch_size, args.max_length)


if __name__ == "__main__":
    main()
