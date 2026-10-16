"""Collect author-code M/B/V reproductions and emit a compact LaTeX table.

Only completed JSON files are included.  The script never fills missing cells
or converts values across evaluation protocols.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def ffmsr_rows() -> list[tuple[str, str, dict[str, float], str]]:
    rows = []
    for result_file in sorted((ROOT / "results" / "ffmsr").glob("*/seed*/results.json")):
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        if payload.get("smoke"):
            continue
        short = {"Music": "M", "Books": "B", "Movies": "V"}
        pair = "--".join(short.get(x, x) for x in payload["domains"])
        budget = (f"{payload.get('pretrain_epochs', 5)} pretrain; "
                  f"{payload.get('finetune_epochs', '?')} fine-tune")
        if "Books" in payload["domains"]:
            budget += (f"; B uses {payload.get('books_bert_layers', '?')} BERT layer, "
                       f"batch {payload.get('books_train_batch_size', '?')}")
        for domain, result in payload["results"].items():
            rows.append(("FFMSR", f"{pair}: {short.get(domain, domain)}",
                         result["test_result"], budget))
    return rows


def feddcsr_rows() -> list[tuple[str, str, dict[str, float], str]]:
    rows = []
    for result_file in sorted((ROOT / "results" / "feddcsr").glob("seed*/FedDCSR_*_results.json")):
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        for domain, result in payload["per_domain"].items():
            metrics = {
                "recall@10": result["HR @10"],
                "recall@20": result["HR @20"],
                "ndcg@10": result["NDCG @10"],
                "ndcg@20": result["NDCG @20"],
            }
            budget = (f"{payload.get('federated_rounds', '?')} rounds; "
                      f"{payload.get('local_epochs', '?')} local epochs")
            rows.append(("FedDCSR", domain, metrics, budget))
    return rows


def latex(rows: list[tuple[str, str, dict[str, float], str]]) -> str:
    lines = [
        r"\begin{table}[t]\centering\tiny",
        r"\caption{Author-code native reproductions on the fixed 5,000-user M/B/V subset (seed 42; full ranking). FFMSR uses 5/20 pretrain/fine-tune epochs for M--V and 1/5 for pairs containing B (final BERT layer; batch 2048); FedDCSR uses 10 rounds $\times$ 3 local epochs.}",
        r"\label{tab:native-reproductions}",
        r"\setlength{\tabcolsep}{1.7pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"System & Run & R@10 & R@20 & N@10 & N@20 \\",
        r"\midrule",
    ]
    for system, domain, metric, budget in rows:
        lines.append(
            f"{system} & {domain} & "
            f"{metric['recall@10']:.4f} & {metric['recall@20']:.4f} & "
            f"{metric['ndcg@10']:.4f} & {metric['ndcg@20']:.4f} \\\\"  # noqa: W605
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = ffmsr_rows() + feddcsr_rows()
    output = ROOT.parent.parent / "paper_www" / "tables_native_reproductions.tex"
    output.write_text(latex(rows), encoding="utf-8")
    print(f"wrote {output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
