"""Chained driver: wait for pretrains, then run cross, indomain, studies."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "outputs" / "www_v1" / "pretrain"


def wait_for_pretrains():
    need = [PRE / f"source_{s}_seed{sd}_adapter.pt" for s in ("music", "books", "movies") for sd in (42, 123, 456, 789, 2024)]
    while True:
        missing = [p for p in need if not p.exists()]
        if not missing:
            return
        print(f"[driver] waiting for {len(missing)} pretrain adapters...", flush=True)
        time.sleep(60)


def count_results(subdir):
    d = ROOT / "outputs" / "www_v1" / subdir
    return len(list(d.rglob("results.json"))) if d.exists() else 0


EXPECTED = {"cross": 16 * 6 * 5, "indomain": 11 * 4 * 5}


def run_stage(stage, max_passes=3):
    print(f"[driver] START {stage} {time.strftime('%H:%M:%S')}", flush=True)
    for p in range(1, max_passes + 1):
        rc = subprocess.call([sys.executable, "-u", "run_www_matrix.py", "--stage", stage], cwd=ROOT)
        n = count_results(stage)
        exp = EXPECTED.get(stage)
        print(f"[driver] pass {p} {stage} rc={rc} results={n}/{exp} {time.strftime('%H:%M:%S')}", flush=True)
        if exp is None or n >= exp:
            break
    print(f"[driver] DONE {stage} {time.strftime('%H:%M:%S')}", flush=True)


wait_for_pretrains()
run_stage("cross")
run_stage("indomain")
run_stage("studies")
run_stage("encoder")
print("[driver] ALL DONE", flush=True)
