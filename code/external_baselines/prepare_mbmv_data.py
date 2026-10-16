"""Prepare the FedPASA Music/Books/Movies data for authors' baseline code.

The script does not alter any baseline model.  It converts the chronologically
ordered Amazon interactions used by FedPASA into:

1. FedDCSR sequence files (one domain is one federated client), and
2. RecBole benchmark files used by FFMSR/PFCR.

By default it uses the same deterministic 5,000-user subset (seed 777) that is
used for cross-domain evaluation in the FedPASA runner.  Pass ``--all-users``
to convert every eligible user.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "amazon_v2_transfr20_30"

DOMAINS = {
    "Music": (
        DATA_ROOT / "cds_and_vinyl" / "cds_and_vinyl.dat",
        DATA_ROOT / "cds_and_vinyl" / "titles_real_metadata.json",
    ),
    "Books": (
        DATA_ROOT / "books" / "books.dat",
        DATA_ROOT / "books" / "titles_real_metadata.json",
    ),
    "Movies": (
        DATA_ROOT / "movies_and_tv" / "movies_and_tv.dat",
        DATA_ROOT / "movies_and_tv" / "titles_real_metadata.json",
    ),
}


def load_sequences(path: Path) -> dict[int, list[int]]:
    rows: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for order, row in enumerate(reader):
            if len(row) < 4:
                continue
            user, item, _, timestamp = row[:4]
            rows[int(user)].append((int(timestamp), order, int(item)))
    out = {}
    for user, events in rows.items():
        if len(events) < 3:
            continue
        events.sort(key=lambda value: (value[0], value[1]))
        out[user] = [item for _, _, item in events]
    return out


def choose_users(sequences: dict[int, list[int]], max_users: int, seed: int) -> list[int]:
    users = sorted(sequences)
    if max_users > 0 and len(users) > max_users:
        users = sorted(random.Random(seed).sample(users, max_users))
    return users


def write_feddcsr(domain: str, sequences: dict[int, list[int]], users: list[int], output: Path) -> None:
    dest = output / domain
    dest.mkdir(parents=True, exist_ok=True)
    observed_items = sorted({item for user in users for item in sequences[user]})
    item_map = {item: index for index, item in enumerate(observed_items)}
    (dest / "num_items.txt").write_text(f"{len(observed_items)}\n", encoding="utf-8")
    with (dest / "item2index.txt").open("w", encoding="utf-8") as map_file:
        for original_item in observed_items:
            map_file.write(f"{original_item}\t{item_map[original_item]}\n")
    with (
        (dest / "train_data.txt").open("w", encoding="utf-8") as train_file,
        (dest / "valid_data.txt").open("w", encoding="utf-8") as valid_file,
        (dest / "test_data.txt").open("w", encoding="utf-8") as test_file,
    ):
        for new_user, old_user in enumerate(users):
            sequence = [item_map[item] for item in sequences[old_user]]
            train = sequence[:-2]
            valid = sequence[-2]
            test = sequence[-1]
            # FedDCSR uses max_seq_len=16.  Each file includes its target item.
            train_seq = train[-17:]
            valid_seq = (train[-16:] + [valid])
            test_seq = ((train + [valid])[-16:] + [test])
            train_file.write(f"{new_user}\t" + "\t".join(map(str, train_seq)) + "\n")
            valid_file.write(f"{new_user}\t" + "\t".join(map(str, valid_seq)) + "\n")
            test_file.write(f"{new_user}\t" + "\t".join(map(str, test_seq)) + "\n")


def write_recbole(
    domain: str,
    sequences: dict[int, list[int]],
    users: list[int],
    titles_path: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / domain
    observed_items = sorted({item for user in users for item in sequences[user]})
    item_map = {item: index for index, item in enumerate(observed_items)}
    with (
        Path(f"{prefix}.train.inter").open("w", encoding="utf-8") as train_file,
        Path(f"{prefix}.valid.inter").open("w", encoding="utf-8") as valid_file,
        Path(f"{prefix}.test.inter").open("w", encoding="utf-8") as test_file,
    ):
        header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
        train_file.write(header)
        valid_file.write(header)
        test_file.write(header)
        for new_user, old_user in enumerate(users):
            sequence = [item_map[item] for item in sequences[old_user]]
            train = sequence[:-2]
            valid = sequence[-2]
            test = sequence[-1]
            for target_idx in range(1, len(train)):
                history = train[:target_idx][-50:]
                train_file.write(
                    f"{new_user}\t{' '.join(map(str, history))}\t{train[target_idx]}\n"
                )
            valid_file.write(
                f"{new_user}\t{' '.join(map(str, train[-50:]))}\t{valid}\n"
            )
            test_history = (train + [valid])[-50:]
            test_file.write(
                f"{new_user}\t{' '.join(map(str, test_history))}\t{test}\n"
            )

    titles = json.loads(titles_path.read_text(encoding="utf-8"))
    with Path(f"{prefix}.text").open("w", encoding="utf-8") as text_file:
        text_file.write("item_id:token\ttext:token_seq\n")
        for original_item in observed_items:
            dense_item = item_map[original_item]
            clean = str(titles.get(str(original_item), f"Item {original_item}")).replace("\t", " ").replace("\n", " ")
            text_file.write(f"{dense_item}\t{clean}\n")
    with Path(f"{prefix}.item2index").open("w", encoding="utf-8") as map_file:
        for original_item in observed_items:
            map_file.write(f"{original_item}\t{item_map[original_item]}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--all-users", action="store_true")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument(
        "--feddcsr-output",
        type=Path,
        default=ROOT / "external_baselines" / "FedDCSR" / "data",
    )
    parser.add_argument(
        "--recbole-output",
        type=Path,
        default=ROOT / "external_baselines" / "recbole_mbmv",
    )
    args = parser.parse_args()
    max_users = 0 if args.all_users else args.max_users
    summary = {}
    for domain, (ratings_path, titles_path) in DOMAINS.items():
        sequences = load_sequences(ratings_path)
        users = choose_users(sequences, max_users, args.seed)
        write_feddcsr(domain, sequences, users, args.feddcsr_output)
        write_recbole(domain, sequences, users, titles_path, args.recbole_output)
        summary[domain] = {
            "users": len(users),
            "items": len({item for user in users for item in sequences[user]}),
            "interactions": sum(len(sequences[user]) for user in users),
            "selection_seed": args.seed,
        }
        print(domain, summary[domain])
    (args.recbole_output / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
