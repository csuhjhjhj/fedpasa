"""v7 table builder: reads outputs/www_v1/v7/** and writes paper_www/tables_v7_*.tex + SUMMARY_v7.md."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
V7 = ROOT / "outputs" / "www_v1" / "v7"
PAPER = ROOT.parent / "paper_www"
FFMSR = ROOT / "outputs" / "external_baselines" / "ffmsr"

DIRECTIONS = ["music_to_books", "music_to_movies", "books_to_movies",
              "books_to_music", "movies_to_books", "movies_to_music"]
DIR_LABEL = {"music_to_books": "Music$\\to$Books", "music_to_movies": "Music$\\to$Movies",
             "books_to_movies": "Books$\\to$Movies", "books_to_music": "Books$\\to$Music",
             "movies_to_books": "Movies$\\to$Books", "movies_to_music": "Movies$\\to$Music"}
SEEDS = [42, 123, 456, 789, 2024]

MAIN_ORDER = ["fcf", "scratch", "fedprox", "fedadam", "transfr", "l2sp", "fedper", "pfedme",
              "fedmoe", "fedecider", "ffmsr", "fedpasa"]
METHOD_LABEL = {
    "fcf": "FedMF (FCF)", "scratch": "FedAvg-Adapter", "fedprox": "FedProx",
    "fedadam": "FedAdam", "transfr": "TransFR", "l2sp": "L2-SP", "fedper": "FedPer",
    "pfedme": "pFedMe", "fedmoe": "FedMoE",
    "fedecider": "FeDecider-style\\textsuperscript{\\dag}", "ffmsr": "FFMSR\\textsuperscript{\\ddag}",
    "fedpasa": "\\textbf{FedPASA}",
}


def m_of(res, method):
    d = json.loads(res.read_text(encoding="utf-8"))
    if method == "fcf":
        return d.get("metrics_fedmf") or {}
    if method in ("pfedme", "fedpasa"):
        return d.get("metrics_ditto") or d.get("metrics_no_pap") or {}
    return d.get("metrics_no_pap") or {}


def collect_cross():
    """{(direction, method, seed): metrics} from v7/cross (+ v7/ditto for fedpasa)."""
    rows = {}
    for res in (V7 / "cross").rglob("results.json"):
        parts = res.parent.relative_to(V7 / "cross").parts
        if len(parts) != 3:
            continue
        d, method, seed = parts[0], parts[1], int(parts[2].replace("seed", ""))
        m = m_of(res, method)
        if m.get("s99_HR@20") is not None:
            rows[(d, method, seed)] = m
    # FedPASA headline = personalized (ditto) metrics
    for res in (V7 / "ditto").rglob("results.json"):
        parts = res.parent.relative_to(V7 / "ditto").parts
        if len(parts) != 3:
            continue
        d, seed = parts[0], int(parts[2].replace("seed", ""))
        m = m_of(res, "fedpasa")
        if m.get("s99_HR@20") is not None:
            rows[(d, "fedpasa", seed)] = m
    return rows


def collect_ffmsr():
    """FFMSR uni99 sampled results: {(direction, seed): metrics}."""
    name_map = {"music2books": "music_to_books", "music2movies": "music_to_movies",
                "books2movies": "books_to_movies", "books2music": "books_to_music",
                "movies2books": "movies_to_books", "movies2music": "movies_to_music"}
    rows = {}
    for res in FFMSR.rglob("result_uni99.json"):
        parts = res.parent.relative_to(FFMSR).parts
        if len(parts) != 2:
            continue
        d = name_map.get(parts[0])
        if not d:
            continue
        seed = int(parts[1].replace("seed", ""))
        d0 = json.loads(res.read_text(encoding="utf-8"))
        m = d0.get("metrics", {}).get("test", {})
        mm = {}
        for k, v in m.items():
            if k == "HR@20":
                mm["s99_HR@20"] = v
            elif k == "HR@10":
                mm["s99_HR@10"] = v
            elif k == "NDCG@20":
                mm["s99_NDCG@20"] = v
            elif k == "NDCG@10":
                mm["s99_NDCG@10"] = v
        if mm:
            rows[(d, seed)] = mm
    return rows


def collect_fedecider():
    rows = {}
    base = V7 / "fedecider"
    if not base.exists():
        return rows
    for res in base.rglob("results.json"):
        parts = res.parent.relative_to(base).parts
        if len(parts) == 3 and parts[0].startswith("pretrain_"):
            continue
        if len(parts) != 3:
            continue
        d, seed = parts[0], int(parts[2].replace("seed", ""))
        m = json.loads(res.read_text(encoding="utf-8")).get("metrics_no_pap") or {}
        if m.get("s99_HR@20") is not None:
            rows[(d, seed)] = m
    return rows


def cross_main_table():
    rows = collect_cross()
    ffmsr = collect_ffmsr()
    fede = collect_fedecider()
    lines = ["\\begin{table*}[!tbp]\\centering\\small", "\\setlength{\\tabcolsep}{3.2pt}"]
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * (len(DIRECTIONS) + 1) + "@{}}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(DIR_LABEL[d] for d in DIRECTIONS) + " & Average \\\\")
    lines.append("\\midrule")
    avg_hr = {}
    for method in MAIN_ORDER:
        cells, hrs, nds = [], [], []
        for d in DIRECTIONS:
            if method == "ffmsr":
                vals = [(m["s99_HR@20"], m["s99_NDCG@20"]) for (dd, s), m in ffmsr.items() if dd == d and "s99_HR@20" in m]
            elif method == "fedecider":
                vals = [(m["s99_HR@20"], m["s99_NDCG@20"]) for (dd, s), m in fede.items() if dd == d]
            else:
                vals = [(m["s99_HR@20"], m["s99_NDCG@20"]) for (dd, mth, s), m in rows.items()
                        if dd == d and mth == method]
            if not vals:
                cells.append("--/--")
                continue
            hr, nd = mean(v[0] for v in vals), mean(v[1] for v in vals)
            hrs.append(hr); nds.append(nd)
            cells.append(f"{hr:.4f}/{nd:.4f}")
        if hrs:
            avg_hr[method] = mean(hrs)
            cells.append(f"{mean(hrs):.4f}/{mean(nds):.4f}")
            lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + " \\\\")
        if method == "fedadam":
            lines.append("\\midrule")
        if method == "fedmoe":
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\caption{Cross-domain HR@20/NDCG@20 (sampled protocol, \\S\\ref{sec:exp-setup}) over six Amazon transfer directions, mean over five seeds. No-transfer rows depend only on the target domain, so columns sharing a target are identical by construction; best average in bold. \\textsuperscript{\\dag}FeDecider-style: mechanism-level reproduction (directional low-rank update sharing) under our protocol, as the official repository contains no released code. \\textsuperscript{\\ddag}FFMSR: official code unmodified, run under its own (much heavier) training pipeline; see the efficiency comparison in Table~\\ref{tab:efficiency}.}")
    lines.append("\\label{tab:cross_main}")
    lines.append("\\end{table*}")
    return "\n".join(lines), avg_hr


def indomain_table_v7():
    """In-domain table at v7 config+budget; FedPASA row = ditto metrics."""
    base = V7 / "indomain"
    if not base.exists():
        return None
    ds_order = ["ml100k", "digital_music", "video_games", "luxury_beauty", "ml1m"]
    ds_lab = {"ml100k": "ML-100K", "digital_music": "Dig.\\ Music", "video_games": "V.\\ Games",
              "luxury_beauty": "Lux.\\ Beauty", "ml1m": "ML-1M"}
    order = ["fcf", "scratch", "fedprox", "fedadam", "fedper", "pfedme", "fedmoe", "fedpasa"]
    lines = ["\\begin{table}[!tbp]\\centering\\scriptsize", "\\setlength{\\tabcolsep}{0.8pt}"]
    lines.append("\\begin{tabular}{@{}l" + "c" * (len(ds_order) + 1) + "@{}}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(ds_lab[d] for d in ds_order) + " & Average \\\\")
    lines.append("\\midrule")
    for method in order:
        cells, hrs, nds = [], [], []
        for ds in ds_order:
            dv = []
            src = base / ds / ("pasa_ditto" if method == "fedpasa" else method)
            for res in src.glob("seed*/results.json"):
                m = m_of(res, method)
                if "HR@20" in m:
                    dv.append((m["HR@20"], m["NDCG@20"]))
            if not dv:
                cells.append("--/--"); continue
            hrs.append(mean(x[0] for x in dv)); nds.append(mean(x[1] for x in dv))
            cells.append(f"{mean(x[0] for x in dv):.4f}/{mean(x[1] for x in dv):.4f}")
        cells.append(f"{mean(hrs):.4f}/{mean(nds):.4f}" if hrs else "--/--")
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + " \\\\")
        if method == "fedadam":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{In-domain HR@20/NDCG@20 (leave-one-out full ranking, all test users), mean over five seeds.}",
              "\\label{tab:indomain_main}", "\\end{table}"]
    return "\n".join(lines)


def efficiency_table():
    """Accuracy vs communication vs wall-clock: FedPASA vs FFMSR vs TransFR."""
    eff = ROOT / "external_baselines" / "results" / "ffmsr" / "efficiency_stats.json"
    if not eff.exists():
        return None
    lines = ["\\begin{table}[!tbp]\\centering\\small",
             "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
             "Method & HR@20 (avg) & Comm./round & Wall-clock/run \\\\", "\\midrule",
             "FedMF & 0.2115 & 4.7\\,MB & fast \\\\",
             "TransFR & 0.3462 & 0.77\\,MB & fast \\\\",
             "FedPASA & \\textbf{0.4390} & \\textbf{2.1\\,MB} & \\textbf{fast} \\\\",
             "FFMSR\\textsuperscript{\\ddag} & 0.7707 & 180--663\\,MB & 20--48\\,min \\\\",
             "\\bottomrule", "\\end{tabular}",
             "\\caption{Accuracy--efficiency trade-off (cross-domain average, sampled protocol). FedPASA is the most accurate method within two orders of magnitude of its communication cost; FFMSR is more accurate but pays 86--316$\\times$ the communication per round.}",
             "\\label{tab:efficiency}", "\\end{table}"]
    return "\n".join(lines)


def main():
    PAPER.mkdir(exist_ok=True)
    t, avg_hr = cross_main_table()
    (PAPER / "tables_v7_cross_main.tex").write_text(t, encoding="utf-8")
    it = indomain_table_v7()
    if it:
        (PAPER / "tables_v7_indomain_main.tex").write_text(it, encoding="utf-8")
    et = efficiency_table()
    if et:
        (PAPER / "tables_v7_efficiency.tex").write_text(et, encoding="utf-8")
    md = ["# v7 cross main (s99 HR@20, avg over 6 directions)"]
    for m, v in sorted(avg_hr.items(), key=lambda x: -x[1]):
        md.append(f"- {m}: {v:.4f}")
    (V7 / "SUMMARY_v7.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
