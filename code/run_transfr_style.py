import argparse
import csv
import hashlib
import importlib.machinery
import json
import math
import random
import sys
import time
import types
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Text-only DistilBERT experiments do not need torchvision. Some local
# torch/torchvision combinations fail during Transformers' optional vision import.
try:
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_io_stub = types.ModuleType("torchvision.io")
    torchvision_transforms_stub = types.ModuleType("torchvision.transforms")
    torchvision_tf_stub = types.ModuleType("torchvision.transforms.functional")
    torchvision_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    torchvision_io_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", loader=None)
    torchvision_transforms_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
    torchvision_tf_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.functional", loader=None)

    class _ImageReadMode:
        RGB = "RGB"

    class _InterpolationMode:
        NEAREST_EXACT = "nearest_exact"
        BOX = "box"
        BILINEAR = "bilinear"
        HAMMING = "hamming"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    torchvision_io_stub.ImageReadMode = _ImageReadMode
    torchvision_io_stub.decode_image = lambda *args, **kwargs: None
    torchvision_transforms_stub.InterpolationMode = _InterpolationMode
    torchvision_tf_stub.pil_to_tensor = lambda image: image
    torchvision_stub.io = torchvision_io_stub
    torchvision_stub.transforms = torchvision_transforms_stub
    sys.modules.setdefault("torchvision", torchvision_stub)
    sys.modules.setdefault("torchvision.io", torchvision_io_stub)
    sys.modules.setdefault("torchvision.transforms", torchvision_transforms_stub)
    sys.modules.setdefault("torchvision.transforms.functional", torchvision_tf_stub)

    import transformers.utils.import_utils as _tf_import_utils
    _tf_import_utils.is_torchvision_available = lambda: False
except Exception:
    pass

from transformers import AutoModel, AutoTokenizer


ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_fedcia_dat(path: Path):
    cache = path.with_suffix(path.suffix + ".npz")
    if cache.exists():
        z = np.load(cache)
        rows = list(zip(z["user"].tolist(), z["item"].tolist(), z["rating"].tolist(), z["ts"].tolist()))
    else:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                parts = raw.split(",") if "," in raw else raw.split()
                if len(parts) < 4:
                    raise ValueError(f"Expected four interaction columns at {path}:{line_number}")
                user, item, rating, ts = parts[:4]
                rows.append((int(user), int(item), float(rating), int(float(ts))))
        try:
            np.savez_compressed(cache,
                                user=np.array([r[0] for r in rows], dtype=np.int64),
                                item=np.array([r[1] for r in rows], dtype=np.int64),
                                rating=np.array([r[2] for r in rows], dtype=np.float32),
                                ts=np.array([r[3] for r in rows], dtype=np.int64))
        except OSError:
            pass
    if not rows:
        return defaultdict(list)
    user_offset = 1 if min(row[0] for row in rows) == 1 else 0
    item_offset = 1 if min(row[1] for row in rows) == 1 else 0
    rows = [
        (user - user_offset, item - item_offset, rating, ts)
        for user, item, rating, ts in rows
    ]
    rows.sort(key=lambda x: (x[0], x[3]))
    by_user = defaultdict(list)
    for user, item, _, ts in rows:
        by_user[user].append((item, ts))
    return by_user


def leave_one_out_split(by_user):
    train, valid, test = {}, {}, {}
    for user, seq in by_user.items():
        items = [item for item, _ in sorted(seq, key=lambda x: x[1])]
        if len(items) < 3:
            continue
        train[user] = items[:-2]
        valid[user] = items[-2]
        test[user] = items[-1]
    return train, valid, test


def ensure_ml100k_titles(cache_dir: Path, max_item_id: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ml-100k.zip"
    extract_dir = cache_dir / "ml-100k"
    item_path = extract_dir / "u.item"
    if not item_path.exists():
        if not zip_path.exists():
            print(f"Downloading MovieLens 100K metadata to {zip_path}")
            urllib.request.urlretrieve(ML100K_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)

    raw_titles = {}
    with item_path.open("r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if not row:
                continue
            movie_id = int(row[0]) - 1
            title = row[1]
            genres = row[5:]
            genre_names = [
                "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
            ]
            active = [name for name, flag in zip(genre_names, genres) if flag == "1"]
            raw_titles[movie_id] = title if not active else f"{title}. Genres: {', '.join(active)}"

    titles = {}
    for item_id in range(max_item_id + 1):
        titles[item_id] = raw_titles.get(item_id, f"movie item {item_id}")
    return titles


def ensure_ml1m_titles(cache_dir: Path, max_item_id: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ml-1m.zip"
    extract_dir = cache_dir / "ml-1m"
    item_path = extract_dir / "movies.dat"
    if not item_path.exists():
        if not zip_path.exists():
            print(f"Downloading MovieLens 1M metadata to {zip_path}")
            urllib.request.urlretrieve(ML1M_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)
    raw_titles = {}
    with item_path.open("r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            movie_id = int(parts[0]) - 1
            raw_titles[movie_id] = f"{parts[1]}. Genres: {parts[2].replace('|', ', ')}"
    return {item_id: raw_titles.get(item_id, f"movie item {item_id}") for item_id in range(max_item_id + 1)}


def fallback_titles(max_item_id: int, dataset_name: str):
    prefix = dataset_name.replace("_", " ")
    return {item_id: f"{prefix} item {item_id}" for item_id in range(max_item_id + 1)}


def load_titles_json(path: Path, max_item_id: int, dataset_name: str):
    raw = json.loads(path.read_text(encoding="utf-8"))
    titles = fallback_titles(max_item_id, dataset_name)
    if isinstance(raw, list):
        for item_id, title in enumerate(raw[: max_item_id + 1]):
            if title:
                titles[item_id] = str(title)
    else:
        for key, title in raw.items():
            item_id = int(key)
            if 0 <= item_id <= max_item_id and title:
                titles[item_id] = str(title)
    return titles


def load_precomputed_text_features(path: Path):
    if path.suffix == ".pt":
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            tensor = tensor["features"]
        tensor = tensor.float()
    else:
        arr = np.load(path)
        tensor = torch.tensor(arr, dtype=torch.float32)
    if tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return F.normalize(tensor, dim=-1)


def build_hash_text_features(titles, dim: int = 768):
    ordered_ids = sorted(titles)
    features = torch.zeros((len(ordered_ids), dim), dtype=torch.float32)
    for row, item_id in enumerate(ordered_ids):
        text = str(titles[item_id]).lower()
        tokens = text.replace("_", " ").split()
        units = tokens + [text[max(0, i - 2): i + 1] for i in range(len(text))]
        if not units:
            units = [str(item_id)]
        for unit in units:
            digest = hashlib.blake2b(unit.encode("utf-8", errors="ignore"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            features[row, bucket] += sign
        features[row] += (int(item_id) % 997) / 9970.0
    return F.normalize(features, dim=-1)


@torch.no_grad()
def build_text_features(
    titles,
    model_name: str,
    cache_path: Path,
    batch_size: int,
    device: torch.device,
    allow_hash_fallback: bool = False,
):
    meta_path = cache_path.with_suffix(".meta.json")
    hasher = hashlib.sha256()
    for item_id in sorted(titles):
        hasher.update(str(item_id).encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(titles[item_id].encode("utf-8"))
        hasher.update(b"\n")
    expected = {
        "model_name": model_name,
        "num_items": len(titles),
        "titles_hash": hasher.hexdigest(),
    }
    if cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            comparable = {key: meta.get(key) for key in expected}
            if comparable == expected:
                if meta.get("fallback") == "hash_text_features_v1" and not allow_hash_fallback:
                    raise RuntimeError(
                        f"Refusing to reuse hashed text-feature cache without --allow_hash_text_fallback: {cache_path}"
                    )
                return torch.load(cache_path, map_location="cpu")
        except Exception:
            if not allow_hash_fallback:
                raise

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    except Exception as exc:
        if not allow_hash_fallback:
            raise RuntimeError(
                f"Failed to load real text encoder '{model_name}'. "
                "Install/cache the model or pass --allow_hash_text_fallback only for non-paper smoke tests."
            ) from exc
        print(f"WARNING: failed to load {model_name}; using deterministic hashed text features. reason={exc}", flush=True)
        features = build_hash_text_features(titles)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, cache_path)
        expected["fallback"] = "hash_text_features_v1"
        meta_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        return features
    model.eval()
    ordered_ids = sorted(titles)
    vectors = []
    for start in tqdm(range(0, len(ordered_ids), batch_size), desc="Encoding item text"):
        batch_ids = ordered_ids[start:start + batch_size]
        texts = [titles[i] for i in batch_ids]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=96, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        vectors.append(F.normalize(pooled.detach().cpu(), dim=-1))
    features = torch.cat(vectors, dim=0)
    torch.save(features, cache_path)
    meta_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return features


class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks = []
        for _ in range(max(1, layers)):
            blocks.extend([
                nn.LayerNorm(dim),
                nn.Linear(dim, bottleneck),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, dim),
            ])
        self.blocks = nn.ModuleList()
        for idx in range(0, len(blocks), 5):
            self.blocks.append(nn.Sequential(*blocks[idx:idx + 5]))

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return F.normalize(x, dim=-1)


class FedBATAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck: int,
        layers: int = 2,
        dropout: float = 0.1,
        alpha: float = 0.35,
        condition_scale: float = 0.25,
        cold_power: float = 1.0,
        cold_threshold: float = 0.0,
        use_cold: bool = True,
        num_items: int = 0,
        anchor_item: bool = False,
        anchor_item_init: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.condition_scale = condition_scale
        self.cold_power = cold_power
        self.cold_threshold = cold_threshold
        self.use_cold = use_cold
        self.warm = Adapter(dim, bottleneck, layers, dropout)
        self.cold = Adapter(dim, bottleneck, layers, dropout)
        self.condition = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim * 2),
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        # Optional per-item learnable anchor gate (federated with the adapter).
        # rho_i = anchor_weight * sigmoid(anchor_gate[i]); initialized so that
        # sigmoid(gate) ~= anchor_item_init (the catalog-level gate value).
        self.anchor_item = bool(anchor_item and num_items > 0)
        if self.anchor_item:
            init_p = min(max(float(anchor_item_init), 1e-4), 1.0 - 1e-4)
            init_logit = math.log(init_p / (1.0 - init_p))
            self.anchor_gate = nn.Parameter(torch.full((num_items,), init_logit))

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def forward(self, x, client_proto=None, coldness=None, cond_scale=None):
        warm = self.warm(x)
        cold = self.cold(x)
        if not self.use_cold:
            coldness = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        elif coldness is None:
            coldness = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        if coldness.dim() == 1:
            coldness = coldness.unsqueeze(-1)
        if self.cold_threshold > 0:
            coldness = ((coldness - self.cold_threshold) / max(1e-6, 1.0 - self.cold_threshold)).clamp(0, 1)
        if self.cold_power != 1.0:
            coldness = coldness.clamp(0, 1).pow(self.cold_power)
        z = coldness * cold + (1 - coldness) * warm
        if cond_scale is None:
            scale = self.condition_scale
        elif torch.is_tensor(cond_scale):
            scale = torch.clamp(cond_scale, 0.0, 1.0)
        else:
            scale = float(cond_scale)
        if client_proto is not None and not (isinstance(scale, float) and scale == 0.0):
            if client_proto.dim() == 1:
                client_proto = client_proto.unsqueeze(0)
            gamma_beta = self.condition(client_proto.to(x.device, dtype=x.dtype))
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = torch.tanh(gamma)
            z = z * (1 + scale * gamma) + scale * beta
        z = (1 - self.alpha) * warm + self.alpha * z
        return F.normalize(z, dim=-1)


class MoEAdapter(nn.Module):
    """Two-expert mixture with a learned input-dependent gate (MoE baseline).

    Unlike popularity-routed dual-path adapters, the gate is a learned linear
    function of the item representation: g(x) = softmax(Wx / tau).
    """

    def __init__(self, dim: int, bottleneck: int, layers: int = 2, dropout: float = 0.1, tau: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        self.expert0 = Adapter(dim, bottleneck, layers, dropout)
        self.expert1 = Adapter(dim, bottleneck, layers, dropout)
        self.gate = nn.Linear(dim, 2)

    def forward(self, x):
        g = torch.softmax(self.gate(x) / max(self.tau, 1e-6), dim=-1)
        z = g[..., 0:1] * self.expert0(x) + g[..., 1:2] * self.expert1(x)
        return F.normalize(z, dim=-1)


class DualAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck: int,
        layers: int = 2,
        dropout: float = 0.1,
        personal_mix: float = 0.5,
    ):
        super().__init__()
        self.personal_mix = personal_mix
        self.global_branch = Adapter(dim, bottleneck, layers, dropout)
        self.personal_branch = Adapter(dim, bottleneck, layers, dropout)

    def forward(self, x):
        global_z = self.global_branch(x)
        personal_z = self.personal_branch(x)
        return F.normalize((1 - self.personal_mix) * global_z + self.personal_mix * personal_z, dim=-1)


class LocalHead(nn.Module):
    def __init__(self, dim: int, bottleneck: int, dropout: float = 0.1, zero_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.normalize(x + self.net(x), dim=-1)


def make_clients(users, num_clients: int, seed: int, train=None, mode: str = "iid", alpha: float = 1.0):
    users = list(users)
    rng = random.Random(seed)
    if mode == "iid" or train is None:
        rng.shuffle(users)
        clients = [[] for _ in range(num_clients)]
        for idx, user in enumerate(users):
            clients[idx % num_clients].append(user)
        return clients

    np_rng = np.random.default_rng(seed)
    if mode == "dirichlet_activity":
        scores = np.array([len(train.get(user, [])) for user in users], dtype=np.float64)
    elif mode == "dirichlet_popularity":
        item_freq = defaultdict(int)
        for seq in train.values():
            for item in seq:
                item_freq[item] += 1
        scores = np.array(
            [
                float(np.mean([item_freq.get(item, 0) for item in train.get(user, [])]))
                if train.get(user) else 0.0
                for user in users
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown client split mode: {mode}")

    clients = [[] for _ in range(num_clients)]
    quantiles = np.quantile(scores, [0.0, 0.25, 0.5, 0.75, 1.0])
    if np.unique(quantiles).size <= 2:
        rng.shuffle(users)
        for idx, user in enumerate(users):
            clients[idx % num_clients].append(user)
        return clients
    bucket_ids = np.digitize(scores, quantiles[1:-1], right=True)
    for bucket in sorted(set(bucket_ids.tolist())):
        bucket_users = [user for user, bid in zip(users, bucket_ids) if bid == bucket]
        np_rng.shuffle(bucket_users)
        probs = np_rng.dirichlet(np.full(num_clients, max(alpha, 1e-3)))
        raw_counts = np.floor(probs * len(bucket_users)).astype(int)
        while raw_counts.sum() < len(bucket_users):
            raw_counts[int(np.argmax(probs - raw_counts / max(1, len(bucket_users))))] += 1
        start = 0
        for cid, count in enumerate(raw_counts):
            if count > 0:
                clients[cid].extend(bucket_users[start : start + count])
            start += count
    for cid, client_users in enumerate(clients):
        if not client_users:
            donor = max(range(num_clients), key=lambda idx: len(clients[idx]))
            if clients[donor]:
                clients[cid].append(clients[donor].pop())
    return clients


def sample_training_batch(client_users, train, num_items, batch_size, negatives, rng):
    users, items, labels = [], [], []
    for _ in range(batch_size):
        user = rng.choice(client_users)
        pos = rng.choice(train[user])
        seen = set(train[user])
        users.append(user)
        items.append(pos)
        labels.append(1.0)
        for _ in range(negatives):
            neg = rng.randrange(num_items)
            while neg in seen:
                neg = rng.randrange(num_items)
            users.append(user)
            items.append(neg)
            labels.append(0.0)
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(items, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def sample_pairwise_batch(client_users, train, num_items, batch_size, negatives, rng):
    users, pos_items, neg_items = [], [], []
    for _ in range(batch_size):
        user = rng.choice(client_users)
        pos = rng.choice(train[user])
        seen = set(train[user])
        for _ in range(negatives):
            neg = rng.randrange(num_items)
            while neg in seen:
                neg = rng.randrange(num_items)
            users.append(user)
            pos_items.append(pos)
            neg_items.append(neg)
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(pos_items, dtype=torch.long),
        torch.tensor(neg_items, dtype=torch.long),
    )


def score_items(
    user_embeddings,
    adapter,
    head,
    text_features,
    users,
    items,
    device,
    args,
    client_proto=None,
    coldness_table=None,
    item_bias=None,
    item_residual=None,
    cond_scale=None,
):
    u = user_embeddings(users.to(device))
    q = text_features[items.to(device)]
    coldness = None
    if coldness_table is not None:
        coldness = coldness_table[items.to(coldness_table.device)].to(device)
    q = item_representation(adapter, q, args, client_proto, coldness, item_ids=items.to(device), cond_scale=cond_scale)
    if item_residual is not None:
        q = F.normalize(q + args.item_residual_scale * item_residual(items.to(device)), dim=-1)
    if head is not None:
        q = head(q)
    scores = (u * q).sum(dim=-1)
    if item_bias is not None:
        scores = scores + item_bias(items.to(device)).squeeze(-1)
    return scores


def client_text_profile(client_users, train, text_features, chunk_size: int = 200_000):
    item_ids = []
    for user in client_users:
        item_ids.extend(train[user])
    if not item_ids:
        return torch.zeros(text_features.size(1))
    ids = torch.tensor(item_ids, dtype=torch.long)
    # Chunked accumulation: avoid materializing a num_interactions x dim gather.
    total = torch.zeros(text_features.size(1), dtype=torch.float32)
    for start in range(0, ids.numel(), chunk_size):
        total += text_features[ids[start:start + chunk_size]].float().sum(dim=0)
    profile = total / ids.numel()
    return F.normalize(profile, dim=0)


def item_coldness(train, num_items):
    counts = np.zeros(num_items, dtype=np.float64)
    users_items = [np.asarray(seq, dtype=np.int64) for seq in train.values() if len(seq)]
    if users_items:
        all_items = np.concatenate(users_items)
        valid_items = all_items[(all_items >= 0) & (all_items < num_items)]
        counts += np.bincount(valid_items, minlength=num_items)
    counts = torch.from_numpy(counts).float()
    cold = 1.0 / torch.sqrt(counts + 1.0)
    if cold.max() > cold.min():
        cold = (cold - cold.min()) / (cold.max() - cold.min())
    return cold


def item_counts(train, num_items):
    counts = np.zeros(num_items, dtype=np.float64)
    users_items = [np.asarray(seq, dtype=np.int64) for seq in train.values() if len(seq)]
    if users_items:
        all_items = np.concatenate(users_items)
        valid_items = all_items[(all_items >= 0) & (all_items < num_items)]
        counts += np.bincount(valid_items, minlength=num_items)
    return torch.from_numpy(counts).float()


def client_popularity_tables(clients, train, num_items):
    tables = []
    for client_users in clients:
        counts = np.zeros(num_items, dtype=np.float64)
        seqs = [np.asarray(train[user], dtype=np.int64) for user in client_users if train.get(user)]
        if seqs:
            all_items = np.concatenate(seqs)
            valid_items = all_items[(all_items >= 0) & (all_items < num_items)]
            counts += np.bincount(valid_items, minlength=num_items)
        scores = torch.from_numpy(np.log1p(counts)).float()
        if scores.max() > 0:
            scores = scores / scores.max()
        tables.append(scores)
    return tables


def resolve_semantic_anchor_weight(args):
    if args.semantic_anchor_auto <= 0:
        return args.semantic_anchor_weight
    density = args.observed_num_users / max(1, args.num_items)
    low = args.semantic_anchor_auto_low
    high = args.semantic_anchor_auto_high
    if high <= low:
        gate = 1.0 if density <= low else 0.0
        return args.semantic_anchor_weight * gate
    gate = max(0.0, min(1.0, (high - density) / (high - low)))
    return args.semantic_anchor_weight * gate


def apply_fedbasa_policy(args):
    """Record paper-method metadata.

    Note: this function must NEVER silently rewrite the method, server
    optimizer, or any hyperparameter. Every reported configuration is exactly
    the one given on the command line (uniform policy across datasets).
    """
    if args.method not in {"fedpasa", "fedbasa"}:
        args.paper_method = args.method
        args.fedbasa_requested = False
        args.fedbasa_resolved_method = args.method
        args.fedbasa_adapter_mode = args.method
        args.fedbasa_policy_note = "not_fedbasa"
        return

    requested_method = args.method
    args.paper_method = "FedPASA" if requested_method == "fedpasa" else "FedBASA"
    args.fedbasa_requested = True
    args.fedbasa_original_method = requested_method
    # Uniform resolution: the paper method is exactly the fedbat adapter with
    # the caller-specified routing/anchor settings on every dataset.
    args.method = "fedbat"
    args.fedbasa_adapter_mode = "popularity_conditioned_dual_path"
    args.fedbasa_policy_note = "uniform_three_module_framework"
    args.fedbasa_server_mode = args.server_optimizer
    args.fedbasa_resolved_method = args.method
    args.fedbasa_semantic_anchor_mode = (
        "auto" if args.semantic_anchor_auto > 0 else "manual_or_off"
    )


def create_adapter(args, dim, device):
    if args.method == "fedbat":
        anchor_item_init = 0.0
        if getattr(args, "anchor_mode", "catalog") == "item":
            base = float(getattr(args, "effective_semantic_anchor_weight", args.semantic_anchor_weight))
            anchor_item_init = base / max(args.semantic_anchor_weight, 1e-6)
        return FedBATAdapter(
            dim,
            args.adapter_dim,
            args.adapter_layers,
            args.dropout,
            args.fedbat_alpha,
            args.fedbat_condition_scale,
            args.fedbat_cold_power,
            args.fedbat_cold_threshold,
            bool(getattr(args, "fedbat_use_cold", 1)),
            num_items=args.num_items,
            anchor_item=(getattr(args, "anchor_mode", "catalog") == "item"),
            anchor_item_init=anchor_item_init,
        ).to(device)
    if args.method == "fedmoe":
        return MoEAdapter(
            dim,
            args.adapter_dim,
            args.adapter_layers,
            args.dropout,
            getattr(args, "fedmoe_tau", 1.0),
        ).to(device)
    if args.method == "dual":
        return DualAdapter(
            dim,
            args.adapter_dim,
            args.adapter_layers,
            args.dropout,
            args.dual_personal_mix,
        ).to(device)
    return Adapter(dim, args.adapter_dim, args.adapter_layers, args.dropout).to(device)


def round_fedbat_alpha(args, round_id: int) -> float:
    if args.dynamic_alpha_mode == "fixed":
        return float(args.fedbat_alpha)
    # Exponential warm-up: early rounds rely more on warm branch, later rounds
    # use the full behavior-aware mixed branch.
    tau = max(float(args.dynamic_alpha_tau), 1e-6)
    return float(args.fedbat_alpha) * (1.0 - math.exp(-float(round_id + 1) / tau))


def set_adapter_alpha(adapter, alpha: float) -> None:
    if hasattr(adapter, "set_alpha"):
        adapter.set_alpha(alpha)


def adapter_forward(adapter, x, args, client_proto=None, coldness=None, cond_scale=None):
    if args.method == "fedbat":
        return adapter(x, client_proto=client_proto, coldness=coldness, cond_scale=cond_scale)
    return adapter(x)


def item_representation(adapter, raw_features, args, client_proto=None, coldness=None, item_ids=None, cond_scale=None):
    adapted = adapter_forward(adapter, raw_features, args, client_proto, coldness, cond_scale=cond_scale)
    # Per-item learnable anchor gate (federated with the adapter).
    if getattr(args, "anchor_mode", "catalog") == "item" and hasattr(adapter, "anchor_gate") and adapter.anchor_item:
        if item_ids is None:
            item_ids = torch.arange(raw_features.size(0), device=raw_features.device)
        gate = torch.sigmoid(adapter.anchor_gate[item_ids.to(adapter.anchor_gate.device)]).to(raw_features.dtype)
        w = (float(args.semantic_anchor_weight) * gate).clamp(0.0, 1.0).unsqueeze(-1)
        raw = F.normalize(raw_features, dim=-1)
        return F.normalize((1 - w) * adapted + w * raw, dim=-1)
    anchor_weight = getattr(args, "effective_semantic_anchor_weight", args.semantic_anchor_weight)
    if anchor_weight <= 0 and args.semantic_anchor_cold_delta <= 0:
        return adapted
    raw = F.normalize(raw_features, dim=-1)
    if args.semantic_anchor_cold_delta != 0 and coldness is not None:
        if coldness.dim() == 1:
            coldness = coldness.unsqueeze(-1)
        anchor_weight = anchor_weight + args.semantic_anchor_cold_delta * coldness.to(raw.device, raw.dtype)
    anchor_weight = torch.as_tensor(anchor_weight, device=raw.device, dtype=raw.dtype).clamp(0.0, 1.0)
    return F.normalize((1 - anchor_weight) * adapted + anchor_weight * raw, dim=-1)


def train_client(
    initial_adapter_state,
    user_emb_state,
    text_features,
    client_users,
    train,
    args,
    device,
    cid,
    round_id,
    client_proto=None,
    coldness_table=None,
    l2sp_reference=None,
    item_bias_state=None,
    item_residual_state=None,
    lambda_state=None,
):
    lambda_c = None
    if getattr(args, "fedbat_learn_lambda", 0) and args.method == "fedbat":
        init_val = float(lambda_state) if lambda_state is not None else float(args.fedbat_condition_scale)
        lambda_c = torch.tensor(init_val, dtype=torch.float32, device=device, requires_grad=True)
    adapter = create_adapter(args, text_features.size(1), device)
    adapter.load_state_dict(initial_adapter_state)
    if args.method == "fedbat":
        set_adapter_alpha(adapter, round_fedbat_alpha(args, round_id))
    prox_reference = {
        key: value.detach().to(device)
        for key, value in initial_adapter_state.items()
    }
    user_embeddings = nn.Embedding(args.num_users, text_features.size(1)).to(device)
    user_embeddings.load_state_dict(user_emb_state)
    item_bias = None
    if getattr(args, "item_bias", 0) > 0:
        item_bias = nn.Embedding(args.num_items, 1).to(device)
        nn.init.zeros_(item_bias.weight)
        if item_bias_state is not None:
            item_bias.load_state_dict(item_bias_state)
    item_residual = None
    if getattr(args, "item_residual", 0) > 0:
        item_residual = nn.Embedding(args.num_items, text_features.size(1)).to(device)
        nn.init.zeros_(item_residual.weight)
        if item_residual_state is not None:
            item_residual.load_state_dict(item_residual_state)
    params = list(adapter.parameters()) + list(user_embeddings.parameters())
    if item_bias is not None:
        params += list(item_bias.parameters())
    if item_residual is not None:
        params += list(item_residual.parameters())
    if lambda_c is not None:
        opt = torch.optim.AdamW(
            [{"params": params}, {"params": [lambda_c], "lr": args.lr * 30.0, "weight_decay": 0.0}],
            lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed * 100000 + round_id * 100 + cid)
    text_features = text_features.to(device)
    adapter.train()
    user_embeddings.train()
    losses = []
    for _ in range(args.local_steps):
        if args.loss == "bpr":
            users, pos_items, neg_items = sample_pairwise_batch(
                client_users, train, args.num_items, args.batch_size, args.negatives, rng
            )
            pos_logits = score_items(
                user_embeddings, adapter, None, text_features, users, pos_items, device, args, client_proto, coldness_table, item_bias, item_residual, cond_scale=lambda_c
            )
            neg_logits = score_items(
                user_embeddings, adapter, None, text_features, users, neg_items, device, args, client_proto, coldness_table, item_bias, item_residual, cond_scale=lambda_c
            )
            loss = -F.logsigmoid((pos_logits - neg_logits) * args.logit_scale).mean()
            users_for_align = users
            pos_items_for_align = pos_items
        else:
            users, items, labels = sample_training_batch(
                client_users, train, args.num_items, args.batch_size, args.negatives, rng
            )
            labels = labels.to(device)
            logits = score_items(
                user_embeddings, adapter, None, text_features, users, items, device, args, client_proto, coldness_table, item_bias, item_residual, cond_scale=lambda_c
            )
            loss = F.binary_cross_entropy_with_logits(logits * args.logit_scale, labels)
            pos_mask = (labels > 0.5).cpu()
            users_for_align = users[pos_mask]
            pos_items_for_align = items[pos_mask]
        if args.align_weight > 0:
            if len(pos_items_for_align) > 0:
                pos_users = users_for_align.to(device)
                pos_items = pos_items_for_align.to(device)
                u_vec = F.normalize(user_embeddings(pos_users), dim=-1)
                q_raw = F.normalize(text_features[pos_items], dim=-1)
                pos_coldness = None
                if coldness_table is not None:
                    pos_coldness = coldness_table[pos_items.to(coldness_table.device)].to(device)
                q_adapt = item_representation(adapter, text_features[pos_items], args, client_proto, pos_coldness, item_ids=pos_items.to(device), cond_scale=lambda_c)
                align_loss = (1 - (u_vec * q_adapt).sum(dim=-1)).mean()
                retain_loss = (1 - (q_raw * q_adapt).sum(dim=-1)).mean()
                loss = loss + args.align_weight * align_loss + args.semantic_retain_weight * retain_loss
        if args.profile_score_weight != 0 and client_proto is not None:
            if len(pos_items_for_align) > 0:
                pos_users = users_for_align.to(device)
                u_vec = F.normalize(user_embeddings(pos_users), dim=-1)
                proto = client_proto.to(device).unsqueeze(0) if client_proto.dim() == 1 else client_proto.to(device)
                profile_loss = (1 - (u_vec * proto).sum(dim=-1)).mean()
                loss = loss + args.profile_score_weight * profile_loss
        if args.prox_mu > 0:
            prox_loss = 0.0
            for name, param in adapter.named_parameters():
                prox_loss = prox_loss + (param - prox_reference[name]).pow(2).sum()
            loss = loss + 0.5 * args.prox_mu * prox_loss
        if args.l2sp_mu > 0 and l2sp_reference is not None:
            l2sp_loss = 0.0
            for name, param in adapter.named_parameters():
                if name in l2sp_reference:
                    l2sp_loss = l2sp_loss + (param - l2sp_reference[name]).pow(2).sum()
            loss = loss + 0.5 * args.l2sp_mu * l2sp_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    bias_state = item_bias.state_dict() if item_bias is not None else None
    residual_state = item_residual.state_dict() if item_residual is not None else None
    # Move states to CPU clones so per-client GPU tables are released immediately.
    adapter_state_cpu = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    user_state_cpu = {k: v.detach().cpu().clone() for k, v in user_embeddings.state_dict().items()}
    if bias_state is not None:
        bias_state = {k: v.detach().cpu().clone() for k, v in bias_state.items()}
    if residual_state is not None:
        residual_state = {k: v.detach().cpu().clone() for k, v in residual_state.items()}
    lambda_out = None
    if lambda_c is not None:
        lambda_out = float(lambda_c.detach().clamp(0.0, 1.0).cpu())
    if device.type == "cuda":
        del adapter, user_embeddings, opt
        torch.cuda.empty_cache()
    return adapter_state_cpu, user_state_cpu, bias_state, residual_state, lambda_out, float(np.mean(losses))


def aggregate_states(states):
    out = {}
    for key in states[0]:
        out[key] = sum(state[key].detach().cpu() for state in states) / len(states)
    return out


def perturb_client_states(client_states, reference_state, noise_scale: float, seed: int):
    """Heuristic update-perturbation robustness probe (NOT differential privacy).

    Each client's adapter delta is perturbed with i.i.d. Gaussian noise whose
    expected norm equals ``noise_scale * ||delta||``.
    """
    if noise_scale <= 0:
        return client_states
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    out = []
    for state in client_states:
        deltas = [(state[k].detach().cpu() - reference_state[k].detach().cpu()) for k in state]
        flat = torch.cat([d.reshape(-1) for d in deltas])
        sigma = noise_scale * float(flat.norm()) / max(1.0, math.sqrt(flat.numel()))
        noisy = {}
        for k, d in zip(state, deltas):
            noise = torch.randn(d.shape, generator=generator, dtype=d.dtype) * sigma
            noisy[k] = reference_state[k].detach().cpu() + d + noise
        out.append(noisy)
    return out


def privatize_client_states(client_states, reference_state, args, round_id: int):
    if args.dp_noise_multiplier <= 0:
        return client_states
    clip_norm = max(float(args.dp_clip_norm), 1e-12)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.dp_seed) + int(round_id) * 1000003)
    noisy_states = []
    for cid, state in enumerate(client_states):
        delta_parts = []
        for key in state:
            delta_parts.append((state[key].detach().cpu() - reference_state[key].detach().cpu()).reshape(-1))
        flat_delta = torch.cat(delta_parts)
        norm = flat_delta.norm().clamp_min(1e-12)
        scale = min(1.0, clip_norm / float(norm))
        noisy_state = {}
        for key in state:
            delta = (state[key].detach().cpu() - reference_state[key].detach().cpu()) * scale
            noise = torch.randn(
                delta.shape,
                generator=generator,
                dtype=delta.dtype,
                device=delta.device,
            ) * (float(args.dp_noise_multiplier) * clip_norm / max(1, len(client_states)))
            noisy_state[key] = reference_state[key].detach().cpu() + delta + noise
        noisy_states.append(noisy_state)
    return noisy_states


def clone_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def init_server_optimizer_state(state):
    return {
        "t": 0,
        "m": {key: torch.zeros_like(value.detach().cpu()) for key, value in state.items()},
        "v": {key: torch.zeros_like(value.detach().cpu()) for key, value in state.items()},
    }


def server_optimizer_step(previous_state, target_state, opt_state, args):
    if args.server_optimizer == "fedavg":
        return clone_state(target_state), opt_state

    opt_state["t"] += 1
    t = opt_state["t"]
    out = {}
    for key in previous_state:
        prev = previous_state[key].detach().cpu()
        delta = target_state[key].detach().cpu() - prev
        opt_state["m"][key] = args.server_beta1 * opt_state["m"][key] + (1 - args.server_beta1) * delta
        if args.server_optimizer == "fedavgm":
            out[key] = prev + args.server_lr * opt_state["m"][key]
            continue
        delta_sq = delta.pow(2)
        if args.server_optimizer == "fedadagrad":
            opt_state["v"][key] = opt_state["v"][key] + delta_sq
            m_hat = opt_state["m"][key] / max(1 - args.server_beta1 ** t, 1e-12)
            v_hat = opt_state["v"][key]
        elif args.server_optimizer == "fedyogi":
            opt_state["v"][key] = opt_state["v"][key] - (1 - args.server_beta2) * delta_sq * torch.sign(
                opt_state["v"][key] - delta_sq
            )
            m_hat = opt_state["m"][key] / max(1 - args.server_beta1 ** t, 1e-12)
            v_hat = opt_state["v"][key] / max(1 - args.server_beta2 ** t, 1e-12)
        else:
            opt_state["v"][key] = args.server_beta2 * opt_state["v"][key] + (1 - args.server_beta2) * delta_sq
            m_hat = opt_state["m"][key] / max(1 - args.server_beta1 ** t, 1e-12)
            v_hat = opt_state["v"][key] / max(1 - args.server_beta2 ** t, 1e-12)
        out[key] = prev + args.server_lr * m_hat / (torch.sqrt(v_hat.clamp_min(0.0)) + args.server_tau)
    return out, opt_state


def dual_global_local_aggregate_states(states):
    out_states = []
    global_keys = [key for key in states[0] if key.startswith("global_branch.")]
    global_avg = {
        key: sum(state[key].detach().cpu() for state in states) / len(states)
        for key in global_keys
    }
    for state in states:
        out = {}
        for key, value in state.items():
            out[key] = global_avg[key].clone() if key in global_avg else value.detach().cpu().clone()
        out_states.append(out)
    return out_states


def personalized_aggregate_states(states, profiles, temperature: float, self_weight: float):
    if len(states) == 1:
        return states
    profile_mat = torch.stack(profiles)
    sim = profile_mat @ profile_mat.t()
    weights = torch.softmax(sim / max(temperature, 1e-6), dim=1)
    if self_weight > 0:
        eye = torch.eye(len(states))
        weights = (1 - self_weight) * weights + self_weight * eye
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    personalized = []
    for cid in range(len(states)):
        out = {}
        for key in states[0]:
            mixed = None
            for sid, state in enumerate(states):
                value = state[key].detach().cpu() * weights[cid, sid].item()
                mixed = value if mixed is None else mixed + value
            out[key] = mixed
        personalized.append(out)
    return personalized


def adaptive_aggregate_states(states, previous_states, profiles, temperature: float, self_weight: float, update_weight: float):
    if len(states) == 1:
        return states
    profile_mat = torch.stack(profiles)
    profile_sim = profile_mat @ profile_mat.t()
    deltas = []
    for state, prev in zip(states, previous_states):
        parts = []
        for key in state:
            parts.append((state[key].detach().cpu() - prev[key].detach().cpu()).flatten())
        delta = torch.cat(parts)
        deltas.append(F.normalize(delta, dim=0))
    delta_mat = torch.stack(deltas)
    update_sim = delta_mat @ delta_mat.t()
    sim = (1 - update_weight) * profile_sim + update_weight * update_sim
    weights = torch.softmax(sim / max(temperature, 1e-6), dim=1)
    if self_weight > 0:
        eye = torch.eye(len(states))
        weights = (1 - self_weight) * weights + self_weight * eye
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    personalized = []
    for cid in range(len(states)):
        out = {}
        for key in states[0]:
            mixed = None
            for sid, state in enumerate(states):
                value = state[key].detach().cpu() * weights[cid, sid].item()
                mixed = value if mixed is None else mixed + value
            out[key] = mixed
        personalized.append(out)
    return personalized


def decoupled_aggregate_states(
    states,
    previous_states,
    profiles,
    temperature: float,
    self_weight: float,
    profile_weight: float,
    magnitude_weight: float,
):
    if len(states) == 1:
        return states
    profile_mat = torch.stack(profiles)
    profile_sim = profile_mat @ profile_mat.t()
    deltas = []
    norms = []
    key_shapes = []
    for state, prev in zip(states, previous_states):
        parts = []
        if not key_shapes:
            key_shapes = [(key, state[key].shape, state[key].numel()) for key in state]
        for key in state:
            parts.append((state[key].detach().cpu() - prev[key].detach().cpu()).flatten())
        delta = torch.cat(parts)
        norm = delta.norm().clamp_min(1e-12)
        deltas.append(delta)
        norms.append(norm)
    directions = torch.stack([delta / norm for delta, norm in zip(deltas, norms)])
    norms = torch.stack(norms)
    direction_sim = directions @ directions.t()
    log_norms = torch.log(norms.clamp_min(1e-12))
    magnitude_compat = torch.exp(-torch.abs(log_norms[:, None] - log_norms[None, :]))
    sim = (1 - profile_weight) * direction_sim + profile_weight * profile_sim
    sim = sim + magnitude_weight * magnitude_compat
    weights = torch.softmax(sim / max(temperature, 1e-6), dim=1)
    if self_weight > 0:
        eye = torch.eye(len(states))
        weights = (1 - self_weight) * weights + self_weight * eye
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    personalized = []
    for cid in range(len(states)):
        mixed_direction = torch.zeros_like(directions[cid])
        mixed_norm = torch.tensor(0.0)
        for sid in range(len(states)):
            w = weights[cid, sid].item()
            mixed_direction = mixed_direction + w * directions[sid]
            mixed_norm = mixed_norm + w * norms[sid]
        mixed_direction = F.normalize(mixed_direction, dim=0)
        mixed_delta = mixed_norm * mixed_direction
        out = {}
        offset = 0
        for key, shape, numel in key_shapes:
            piece = mixed_delta[offset:offset + numel].reshape(shape)
            out[key] = previous_states[cid][key].detach().cpu() + piece
            offset += numel
        personalized.append(out)
    return personalized


def train_local_heads(global_adapter, local_user_states, text_features, clients, train, args, device, adapter_states=None, client_profiles=None, coldness_table=None):
    heads = []
    text_features = text_features.to(device)
    for cid, client_users in enumerate(clients):
        head = LocalHead(text_features.size(1), args.head_dim, args.dropout, args.head_zero_init).to(device)
        user_embeddings = nn.Embedding(args.num_users, text_features.size(1)).to(device)
        user_embeddings.load_state_dict(local_user_states[cid])
        adapter = create_adapter(args, text_features.size(1), device)
        adapter.load_state_dict(adapter_states[cid] if adapter_states is not None else global_adapter.state_dict())
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False
        for p in user_embeddings.parameters():
            p.requires_grad = False
        opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
        rng = random.Random(args.seed * 700000 + cid)
        head.train()
        for _ in range(args.head_steps):
            proto = client_profiles[cid].to(device) if client_profiles is not None else None
            if args.loss == "bpr":
                users, pos_items, neg_items = sample_pairwise_batch(
                    client_users, train, args.num_items, args.batch_size, args.negatives, rng
                )
                pos_logits = score_items(
                    user_embeddings, adapter, head, text_features, users, pos_items, device, args, proto, coldness_table
                )
                neg_logits = score_items(
                    user_embeddings, adapter, head, text_features, users, neg_items, device, args, proto, coldness_table
                )
                loss = -F.logsigmoid((pos_logits - neg_logits) * args.logit_scale).mean()
            else:
                users, items, labels = sample_training_batch(
                    client_users, train, args.num_items, args.batch_size, args.negatives, rng
                )
                labels = labels.to(device)
                logits = score_items(
                    user_embeddings, adapter, head, text_features, users, items, device, args, proto, coldness_table
                )
                loss = F.binary_cross_entropy_with_logits(logits * args.logit_scale, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        heads.append(head.cpu())
    return heads


@torch.no_grad()
def evaluate(
    global_adapter,
    local_user_states,
    heads,
    text_features,
    clients,
    train,
    valid,
    test,
    args,
    device,
    adapter_states=None,
    client_profiles=None,
    coldness_table=None,
    popularity_counts=None,
    client_popularity=None,
    global_item_bias_state=None,
    item_bias_states=None,
    global_item_residual_state=None,
    item_residual_states=None,
    target_split=None,
    client_lambdas=None,
):
    user_to_client = {}
    for cid, users in enumerate(clients):
        for u in users:
            user_to_client[u] = cid
    item_bases = []
    text_features_device = text_features.to(device)
    coldness_device = coldness_table.to(device) if coldness_table is not None else None
    for cid in range(len(clients)):
        adapter = create_adapter(args, text_features.size(1), device)
        adapter.load_state_dict(adapter_states[cid] if adapter_states is not None else global_adapter.state_dict())
        adapter.eval()
        proto = client_profiles[cid].to(device) if client_profiles is not None else None
        all_item_ids = torch.arange(text_features_device.size(0), device=device)
        lam = client_lambdas[cid] if client_lambdas is not None else None
        item_bases.append(item_representation(adapter, text_features_device, args, proto, coldness_device, item_ids=all_item_ids, cond_scale=lam))
    profile_scores = None
    if client_profiles is not None and args.profile_score_weight != 0:
        profile_scores = []
        for cid in range(len(clients)):
            profile = F.normalize(client_profiles[cid].to(device), dim=0)
            profile_scores.append(torch.matmul(text_features_device, profile))
    global_popularity_scores = None
    if popularity_counts is not None and getattr(args, "global_pop_weight", 0.0) != 0:
        global_popularity_scores = torch.log1p(popularity_counts.to(device))
        if global_popularity_scores.max() > 0:
            global_popularity_scores = global_popularity_scores / global_popularity_scores.max()
    popularity_scores = None
    if client_popularity is not None and args.local_pop_weight != 0:
        popularity_scores = [table.to(device) for table in client_popularity]
    user_models = []
    item_bias_models = []
    item_residual_models = []
    for cid in range(len(clients)):
        # Keep user tables on CPU to bound GPU memory on large domains;
        # batch lookups are cheap and moved per chunk.
        user_embeddings = nn.Embedding(args.num_users, text_features.size(1))
        user_embeddings.load_state_dict(local_user_states[cid])
        user_embeddings.eval()
        user_models.append(user_embeddings)
        bias_model = None
        if getattr(args, "item_bias", 0) > 0:
            bias_model = nn.Embedding(args.num_items, 1).to(device)
            nn.init.zeros_(bias_model.weight)
            state = item_bias_states[cid] if item_bias_states is not None else global_item_bias_state
            if state is not None:
                bias_model.load_state_dict(state)
            bias_model.eval()
        item_bias_models.append(bias_model)
        residual_model = None
        if getattr(args, "item_residual", 0) > 0:
            residual_model = nn.Embedding(args.num_items, text_features.size(1)).to(device)
            nn.init.zeros_(residual_model.weight)
            state = item_residual_states[cid] if item_residual_states is not None else global_item_residual_state
            if state is not None:
                residual_model.load_state_dict(state)
            residual_model.eval()
        item_residual_models.append(residual_model)

    metrics = defaultdict(float)
    sampled_metrics = defaultdict(float)
    count = 0
    cold_metrics = defaultdict(float)
    cold_count = 0
    warm_metrics = defaultdict(float)
    warm_count = 0
    freq_bucket_metrics = {name: defaultdict(float) for name in ("zero", "1_5", "6_20", "20plus")}
    freq_bucket_counts = {name: 0 for name in freq_bucket_metrics}
    act_bucket_metrics = {name: defaultdict(float) for name in ("le10", "11_50", "50plus")}
    act_bucket_counts = {name: 0 for name in act_bucket_metrics}
    conflict_metrics = {name: defaultdict(float) for name in ("conflict", "agree_cold", "warm")}
    conflict_counts = {name: 0 for name in conflict_metrics}
    # Sampled-rank mirrors of the slices (used where full ranking lacks resolution).
    s99_freq_bucket_metrics = {name: defaultdict(float) for name in ("zero", "1_5", "6_20", "20plus")}
    s99_conflict_metrics = {name: defaultdict(float) for name in ("conflict", "agree_cold", "warm")}
    target_split = test if target_split is None else target_split
    eval_users = [u for u in sorted(target_split) if u in train]
    if args.eval_users > 0 and len(eval_users) > args.eval_users:
        # Seeded uniform subsample (identical across methods for paired comparison).
        sub_rng = random.Random(777)
        eval_users = sorted(sub_rng.sample(eval_users, args.eval_users))
    eval_rng = random.Random(args.seed * 7 + 13)  # fixed RNG for reproducible sampled eval
    n_sampled = getattr(args, "eval_sampled_negatives", 0)

    # Representation drift: mean cosine distance between the adapted and the raw
    # frozen item vectors, averaged over clients (pre-head, post-anchoring).
    raw_norm = F.normalize(text_features_device, dim=-1)
    drift_vals = []
    drift_cold_vals = []
    drift_warm_vals = []
    for cid in range(len(clients)):
        dvec = 1.0 - (item_bases[cid] * raw_norm).sum(dim=-1)
        drift_vals.append(float(dvec.mean().item()))
        if popularity_counts is not None:
            cold_mask = popularity_counts.to(device) <= args.cold_item_threshold
            if cold_mask.any():
                drift_cold_vals.append(float(dvec[cold_mask].mean().item()))
            if (~cold_mask).any():
                drift_warm_vals.append(float(dvec[~cold_mask].mean().item()))

    # Per-client raw item counts for global/local popularity conflict slices.
    client_item_counts = None
    if clients is not None:
        client_item_counts = []
        for client_users in clients:
            counts = np.zeros(args.num_items, dtype=np.float64)
            seqs = [np.asarray(train[user], dtype=np.int64) for user in client_users if train.get(user)]
            if seqs:
                all_items = np.concatenate(seqs)
                valid_items = all_items[(all_items >= 0) & (all_items < args.num_items)]
                counts += np.bincount(valid_items, minlength=args.num_items)
            client_item_counts.append(torch.from_numpy(counts).float())

    # ---- batched evaluation: group eval users by client and score in chunks ----
    item_arange = torch.arange(args.num_items, device=device)
    item_vecs_final = []
    for cid in range(len(clients)):
        item_vecs = item_bases[cid]
        if item_residual_models[cid] is not None:
            item_vecs = F.normalize(item_vecs + args.item_residual_scale * item_residual_models[cid](item_arange), dim=-1)
        if heads is not None:
            head = heads[cid].to(device).eval()
            item_vecs = head(item_vecs)
        item_vecs_final.append(item_vecs)
    extra_scores = []
    for cid in range(len(clients)):
        add = None
        if item_bias_models[cid] is not None:
            add = item_bias_models[cid](item_arange).squeeze(-1)
        if profile_scores is not None:
            v = args.profile_score_weight * profile_scores[cid]
            add = v if add is None else add + v
        if global_popularity_scores is not None:
            v = args.global_pop_weight * global_popularity_scores
            add = v if add is None else add + v
        if popularity_scores is not None:
            v = args.local_pop_weight * popularity_scores[cid]
            add = v if add is None else add + v
        if args.cold_score_weight != 0 and coldness_device is not None:
            v = args.cold_score_weight * coldness_device
            add = v if add is None else add + v
        extra_scores.append(add)

    seen_tensors = {}
    for user in eval_users:
        seen_list = list(train[user])
        if target_split is test and user in valid:
            seen_list = seen_list + [valid[user]]
        seen_list = [i for i in seen_list if 0 <= i < args.num_items]
        seen_tensors[user] = torch.tensor(seen_list, dtype=torch.long, device=device)

    sample_gen = None
    if n_sampled > 0:
        sample_gen = torch.Generator().manual_seed(args.seed * 7 + 13)

    # Pre-create metric keys so empty regimes still emit explicit zeros.
    for k in args.metric_ks:
        metrics[f"HR@{k}"] += 0.0
        metrics[f"NDCG@{k}"] += 0.0
        sampled_metrics[f"HR@{k}"] += 0.0
        sampled_metrics[f"NDCG@{k}"] += 0.0

    def record(user, cid, r, r_s):
        nonlocal count, cold_count, warm_count
        target = target_split[user]
        for k in args.metric_ks:
            if r <= k:
                metrics[f"HR@{k}"] += 1.0
                metrics[f"NDCG@{k}"] += 1.0 / math.log2(r + 1)
            if r_s is not None and r_s <= k:
                sampled_metrics[f"HR@{k}"] += 1.0
                sampled_metrics[f"NDCG@{k}"] += 1.0 / math.log2(r_s + 1)
        hist_len = len(train[user])
        if hist_len <= 10:
            act_bucket = "le10"
        elif hist_len <= 50:
            act_bucket = "11_50"
        else:
            act_bucket = "50plus"
        for k in args.metric_ks:
            if r <= k:
                act_bucket_metrics[act_bucket][f"HR@{k}"] += 1.0
                act_bucket_metrics[act_bucket][f"NDCG@{k}"] += 1.0 / math.log2(r + 1)
        act_bucket_counts[act_bucket] += 1
        if popularity_counts is not None and args.cold_item_threshold >= 0:
            target_freq = int(popularity_counts[target].item())
            target_is_cold = target_freq <= args.cold_item_threshold
            bucket_metrics = cold_metrics if target_is_cold else warm_metrics
            for k in args.metric_ks:
                if r <= k:
                    bucket_metrics[f"HR@{k}"] += 1.0
                    bucket_metrics[f"NDCG@{k}"] += 1.0 / math.log2(r + 1)
            if target_is_cold:
                cold_count += 1
            else:
                warm_count += 1
            if target_freq == 0:
                freq_bucket = "zero"
            elif target_freq <= 5:
                freq_bucket = "1_5"
            elif target_freq <= 20:
                freq_bucket = "6_20"
            else:
                freq_bucket = "20plus"
            for k in args.metric_ks:
                if r <= k:
                    freq_bucket_metrics[freq_bucket][f"HR@{k}"] += 1.0
                    freq_bucket_metrics[freq_bucket][f"NDCG@{k}"] += 1.0 / math.log2(r + 1)
                if r_s is not None and r_s <= k:
                    s99_freq_bucket_metrics[freq_bucket][f"HR@{k}"] += 1.0
                    s99_freq_bucket_metrics[freq_bucket][f"NDCG@{k}"] += 1.0 / math.log2(r_s + 1)
            freq_bucket_counts[freq_bucket] += 1
            if client_item_counts is not None:
                local_cnt = float(client_item_counts[cid][target].item())
                if target_is_cold and local_cnt > 0:
                    conflict_bucket = "conflict"
                elif target_is_cold:
                    conflict_bucket = "agree_cold"
                else:
                    conflict_bucket = "warm"
                for k in args.metric_ks:
                    if r <= k:
                        conflict_metrics[conflict_bucket][f"HR@{k}"] += 1.0
                        conflict_metrics[conflict_bucket][f"NDCG@{k}"] += 1.0 / math.log2(r + 1)
                    if r_s is not None and r_s <= k:
                        s99_conflict_metrics[conflict_bucket][f"HR@{k}"] += 1.0
                        s99_conflict_metrics[conflict_bucket][f"NDCG@{k}"] += 1.0 / math.log2(r_s + 1)
                conflict_counts[conflict_bucket] += 1
        count += 1

    batch_size_eval = 256
    for cid in range(len(clients)):
        users_c = [u for u in eval_users if user_to_client.get(u, 0) == cid]
        if not users_c:
            continue
        IVt = item_vecs_final[cid].t().contiguous()
        add = extra_scores[cid]
        for start in tqdm(range(0, len(users_c), batch_size_eval), desc=f"Evaluating client {cid}"):
            batch_users = users_c[start:start + batch_size_eval]
            U = user_models[cid](torch.tensor(batch_users, dtype=torch.long)).to(device)
            sc = U @ IVt
            if add is not None:
                sc = sc + add.unsqueeze(0)
            targets = torch.tensor([target_split[u] for u in batch_users], dtype=torch.long, device=device)
            full = sc.clone()
            for j, u in enumerate(batch_users):
                st = seen_tensors[u]
                if st.numel():
                    full[j].index_fill_(0, st, -1e9)
            tgt_score = full.gather(1, targets.unsqueeze(1))
            ranks = 1 + (full > tgt_score).sum(1)
            ranks_list = ranks.tolist()
            rs_list = [None] * len(batch_users)
            if n_sampled > 0:
                over = max(n_sampled * 3, n_sampled + 16)
                cand = torch.randint(0, args.num_items, (len(batch_users), over), generator=sample_gen)
                for j, u in enumerate(batch_users):
                    cj = cand[j].to(device)
                    keep = cj[~torch.isin(cj, seen_tensors[u]) & (cj != targets[j])]
                    tries = 0
                    while keep.numel() < n_sampled and tries < 5:
                        extra = torch.randint(0, args.num_items, (over,), generator=sample_gen).to(device)
                        extra = extra[~torch.isin(extra, seen_tensors[u]) & (extra != targets[j])]
                        keep = torch.cat([keep, extra])
                        tries += 1
                    keep = keep[:n_sampled]
                    if keep.numel() == 0:
                        continue
                    neg_scores = sc[j][keep]
                    t_score = sc[j][targets[j]]
                    rs_list[j] = int(1 + (neg_scores > t_score).sum().item())
            for j, u in enumerate(batch_users):
                record(u, cid, ranks_list[j], rs_list[j])

    result = {k: v / max(1, count) for k, v in metrics.items()} | {"count": count}
    for k, v in sampled_metrics.items():
        result[f"s99_{k}"] = v / max(1, count)
    result["drift_cos"] = float(np.mean(drift_vals)) if drift_vals else 0.0
    if drift_cold_vals:
        result["drift_cos_cold"] = float(np.mean(drift_cold_vals))
    if drift_warm_vals:
        result["drift_cos_warm"] = float(np.mean(drift_warm_vals))
    for bucket_name, bucket_values in act_bucket_metrics.items():
        bucket_count = act_bucket_counts[bucket_name]
        result[f"uact_{bucket_name}_count"] = bucket_count
        for k, v in bucket_values.items():
            result[f"uact_{bucket_name}_{k}"] = v / max(1, bucket_count)
    if popularity_counts is not None and args.cold_item_threshold >= 0:
        for k, v in cold_metrics.items():
            result[f"cold_{k}"] = v / max(1, cold_count)
        for k, v in warm_metrics.items():
            result[f"warm_{k}"] = v / max(1, warm_count)
        result["cold_count"] = cold_count
        result["warm_count"] = warm_count
        for bucket_name, bucket_values in freq_bucket_metrics.items():
            bucket_count = freq_bucket_counts[bucket_name]
            result[f"freq_{bucket_name}_count"] = bucket_count
            for k, v in bucket_values.items():
                result[f"freq_{bucket_name}_{k}"] = v / max(1, bucket_count)
        for bucket_name, bucket_values in s99_freq_bucket_metrics.items():
            bucket_count = freq_bucket_counts[bucket_name]
            for k, v in bucket_values.items():
                result[f"s99freq_{bucket_name}_{k}"] = v / max(1, bucket_count)
        for bucket_name, bucket_values in conflict_metrics.items():
            bucket_count = conflict_counts[bucket_name]
            result[f"conflict_{bucket_name}_count"] = bucket_count
            for k, v in bucket_values.items():
                result[f"conflict_{bucket_name}_{k}"] = v / max(1, bucket_count)
        for bucket_name, bucket_values in s99_conflict_metrics.items():
            bucket_count = conflict_counts[bucket_name]
            for k, v in bucket_values.items():
                result[f"s99conflict_{bucket_name}_{k}"] = v / max(1, bucket_count)
    return result


def train_eval_mf_baseline(train, valid, test, clients, args, device):
    user_to_client = {}
    for cid, users in enumerate(clients):
        for u in users:
            user_to_client[u] = cid
    user_embs = [
        nn.Embedding(args.num_users, args.mf_dim).to(device)
        for _ in clients
    ]
    item_emb = nn.Embedding(args.num_items, args.mf_dim).to(device)
    for emb in user_embs:
        nn.init.normal_(emb.weight, std=0.02)
    nn.init.normal_(item_emb.weight, std=0.02)
    rng = random.Random(args.seed * 900000)
    for _round in range(args.mf_rounds):
        item_states = []
        for cid, client_users in enumerate(clients):
            local_item = nn.Embedding(args.num_items, args.mf_dim).to(device)
            local_item.load_state_dict(item_emb.state_dict())
            opt = torch.optim.AdamW(
                list(user_embs[cid].parameters()) + list(local_item.parameters()),
                lr=args.mf_lr,
                weight_decay=args.weight_decay,
            )
            for _ in range(args.mf_local_steps):
                users, items, labels = sample_training_batch(
                    client_users, train, args.num_items, args.batch_size, args.negatives, rng
                )
                users = users.to(device)
                items = items.to(device)
                labels = labels.to(device)
                logits = (user_embs[cid](users) * local_item(items)).sum(dim=-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            item_states.append(local_item.state_dict())
        item_emb.load_state_dict(aggregate_states(item_states))

    with torch.no_grad():
        metrics = defaultdict(float)
        for k in args.metric_ks:
            metrics[f"HR@{k}"] += 0.0
            metrics[f"NDCG@{k}"] += 0.0
        count = 0
        eval_users = [u for u in sorted(test) if u in train]
        if args.eval_users > 0 and len(eval_users) > args.eval_users:
            sub_rng = random.Random(777)
            eval_users = sorted(sub_rng.sample(eval_users, args.eval_users))
        all_items = item_emb.weight
        n_sampled = getattr(args, "eval_sampled_negatives", 0)
        sample_gen = torch.Generator().manual_seed(args.seed * 7 + 13) if n_sampled > 0 else None
        sampled = defaultdict(float)
        for k in args.metric_ks:
            sampled[f"HR@{k}"] += 0.0
            sampled[f"NDCG@{k}"] += 0.0
        for user in eval_users:
            cid = user_to_client.get(user, 0)
            scores = torch.matmul(all_items, user_embs[cid](torch.tensor([user], device=device)).squeeze(0))
            seen = set(train[user])
            if user in valid:
                seen.add(valid[user])
            target = test[user]
            if n_sampled > 0:
                over = max(n_sampled * 3, n_sampled + 16)
                cand = torch.randint(0, args.num_items, (over,), generator=sample_gen)
                keep = cand[[(int(c) not in seen and int(c) != target) for c in cand]]
                tries = 0
                while keep.numel() < n_sampled and tries < 5:
                    extra = torch.randint(0, args.num_items, (over,), generator=sample_gen)
                    extra = extra[[(int(c) not in seen and int(c) != target) for c in extra]]
                    keep = torch.cat([keep, extra])
                    tries += 1
                keep = keep[:n_sampled].to(device)
                if keep.numel() > 0:
                    t_score = scores[target]
                    r_s = int(1 + (scores[keep] > t_score).sum().item())
                    for k in args.metric_ks:
                        if r_s <= k:
                            sampled[f"HR@{k}"] += 1.0
                            sampled[f"NDCG@{k}"] += 1.0 / math.log2(r_s + 1)
            if seen:
                seen_idx = torch.tensor([i for i in seen if 0 <= i < args.num_items], device=device)
                scores[seen_idx] = -1e9
            rank = torch.argsort(scores, descending=True)
            pos = (rank == target).nonzero(as_tuple=False)
            if pos.numel() == 0:
                continue
            r = int(pos[0, 0].item()) + 1
            for k in args.metric_ks:
                if r <= k:
                    metrics[f"HR@{k}"] += 1.0
                    metrics[f"NDCG@{k}"] += 1.0 / math.log2(r + 1)
            count += 1
    out = {k: v / max(1, count) for k, v in metrics.items()} | {"count": count}
    for k, v in sampled.items():
        out[f"s99_{k}"] = v / max(1, count)
    return out


def ditto_personalize(global_state, local_user_states, text_features, clients, train, args, device, client_profiles, coldness_table):
    old_steps = args.local_steps
    old_prox = args.prox_mu
    args.local_steps = args.ditto_steps
    args.prox_mu = args.ditto_mu
    personalized_states = []
    personalized_user_states = []
    losses = []
    for cid, client_users in enumerate(tqdm(clients, desc="Ditto personalization")):
        c_state, u_state, _b_state, _r_state, _l, loss = train_client(
            global_state,
            local_user_states[cid],
            text_features,
            client_users,
            train,
            args,
            device,
            cid,
            100000 + cid,
            client_profiles[cid].to(device),
            coldness_table,
        )
        personalized_states.append(c_state)
        personalized_user_states.append(u_state)
        losses.append(loss)
    args.local_steps = old_steps
    args.prox_mu = old_prox
    return personalized_states, personalized_user_states, float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default=Path("data/fedcia_ml100k_100/ml100k_100.dat"))
    parser.add_argument("--output_dir", type=Path, default=Path("transfr_repro/outputs/ml100k_smoke"))
    parser.add_argument("--text_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--title_source", choices=["ml100k", "ml1m", "fallback"], default="ml100k")
    parser.add_argument("--item_titles_json", type=Path, default=None)
    parser.add_argument("--dataset_name", type=str, default="ml100k")
    parser.add_argument("--precomputed_text_features", type=Path, default=None)
    parser.add_argument("--allow_hash_text_fallback", action="store_true")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--client_split", choices=["iid", "dirichlet_activity", "dirichlet_popularity"], default="iid")
    parser.add_argument("--client_dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--method", choices=["transfr", "fedbat", "dual", "fedmoe", "fedbasa", "fedpasa"], default="transfr")
    parser.add_argument("--fedbat_alpha", type=float, default=0.35)
    parser.add_argument("--fedbat_condition_scale", type=float, default=0.25)
    parser.add_argument("--fedbat_cold_power", type=float, default=1.0)
    parser.add_argument("--fedbat_cold_threshold", type=float, default=0.0)
    parser.add_argument("--fedbat_use_cold", type=int, default=1, help="Set 0 to ablate the cold branch (warm-only).")
    parser.add_argument("--fedbat_learn_lambda", type=int, default=0,
                        help="Learn a per-client FiLM conditioning scale lambda_c in [0,1] (local, never federated).")
    parser.add_argument("--fedbat_routing", choices=["popularity", "constant", "random"], default="popularity",
                        help="popularity: evidence-based coldness routing; constant: all items share the mean "
                             "coldness (parameter-matched control); random: coldness scores permuted across items.")
    parser.add_argument("--anchor_mode", choices=["catalog", "item"], default="catalog",
                        help="catalog: domain-density gate; item: per-item learnable federated anchor gate.")
    parser.add_argument("--fedmoe_tau", type=float, default=1.0)
    parser.add_argument("--client_participation", type=float, default=1.0,
                        help="Fraction of clients sampled per round (partial participation).")
    parser.add_argument("--update_noise", type=float, default=0.0,
                        help="Relative Gaussian perturbation on client adapter deltas before aggregation "
                             "(robustness study): noise_norm = update_noise * ||delta||.")
    parser.add_argument("--dual_personal_mix", type=float, default=0.5)
    parser.add_argument("--dual_personal_aggregation", choices=["fedavg", "local"], default="local")
    parser.add_argument("--cold_item_threshold", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local_steps", type=int, default=40)
    parser.add_argument("--head_steps", type=int, default=80)
    parser.add_argument("--head_zero_init", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--loss", choices=["bce", "bpr"], default="bce")
    parser.add_argument("--adapter_dim", type=int, default=64)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--adapter_layers", type=int, default=2)
    parser.add_argument("--item_bias", type=int, default=0, help="Train and federate a lightweight scalar item-bias residual.")
    parser.add_argument("--item_residual", type=int, default=0, help="Train and federate a target-domain item residual embedding table.")
    parser.add_argument("--item_residual_scale", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--logit_scale", type=float, default=10.0)
    parser.add_argument("--aggregation", choices=["fedavg", "semantic", "adaptive", "directional", "decoupled", "local"], default="fedavg")
    parser.add_argument("--server_optimizer", choices=["fedavg", "fedavgm", "fedadam", "fedyogi", "fedadagrad"], default="fedavg")
    parser.add_argument("--server_lr", type=float, default=1.0)
    parser.add_argument("--server_beta1", type=float, default=0.9)
    parser.add_argument("--server_beta2", type=float, default=0.99)
    parser.add_argument("--server_tau", type=float, default=1e-3)
    parser.add_argument(
        "--swa_tail_rounds",
        type=int,
        default=0,
        help="If >0, evaluate the average of the last N round states for adapter/user embeddings.",
    )
    parser.add_argument("--prox_mu", type=float, default=0.0)
    parser.add_argument("--prox_auto", type=int, default=0,
                        help="Density-gate the proximal term like AutoAnchor: prox_eff = prox_mu * clip((eta-low)/(high-low),0,1).")
    parser.add_argument("--dp_noise_multiplier", type=float, default=0.0)
    parser.add_argument("--dp_clip_norm", type=float, default=1.0)
    parser.add_argument("--dp_seed", type=int, default=1729)
    parser.add_argument("--ditto_steps", type=int, default=0)
    parser.add_argument("--ditto_mu", type=float, default=0.1)
    parser.add_argument("--semantic_temperature", type=float, default=0.1)
    parser.add_argument("--semantic_self_weight", type=float, default=0.2)
    parser.add_argument("--adaptive_update_weight", type=float, default=0.5)
    parser.add_argument("--decoupled_profile_weight", type=float, default=0.2)
    parser.add_argument("--decoupled_magnitude_weight", type=float, default=0.2)
    parser.add_argument("--profile_score_weight", type=float, default=0.0)
    parser.add_argument("--global_pop_weight", type=float, default=0.0)
    parser.add_argument("--local_pop_weight", type=float, default=0.0)
    parser.add_argument("--cold_score_weight", type=float, default=0.0)
    parser.add_argument("--align_weight", type=float, default=0.0)
    parser.add_argument("--semantic_retain_weight", type=float, default=0.0)
    parser.add_argument("--semantic_anchor_weight", type=float, default=0.0)
    parser.add_argument("--semantic_anchor_cold_delta", type=float, default=0.0)
    parser.add_argument("--semantic_anchor_auto", type=int, default=0)
    parser.add_argument("--semantic_anchor_auto_low", type=float, default=1.0)
    parser.add_argument("--semantic_anchor_auto_high", type=float, default=1.5)
    parser.add_argument("--dynamic_alpha_mode", choices=["fixed", "exp"], default="fixed")
    parser.add_argument("--dynamic_alpha_tau", type=float, default=5.0)
    parser.add_argument("--l2sp_mu", type=float, default=0.0)
    parser.add_argument("--l2sp_path", type=Path, default=None)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--eval_users", type=int, default=300)
    parser.add_argument("--eval_sampled_negatives", type=int, default=0, help="If >0, use sampled evaluation: rank positive against N random negatives instead of full ranking.")
    parser.add_argument(
        "--train_with_valid_for_test",
        action="store_true",
        help="Use all non-test user interactions for final test training. Validation metrics are skipped because the validation item is included in training.",
    )
    parser.add_argument("--skip_valid_eval", action="store_true", help="Skip validation-set evaluation after test evaluation.")
    parser.add_argument("--metric_ks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--run_mf_baseline", action="store_true")
    parser.add_argument("--only_mf_baseline", action="store_true")
    parser.add_argument("--mf_dim", type=int, default=64)
    parser.add_argument("--mf_rounds", type=int, default=5)
    parser.add_argument("--mf_local_steps", type=int, default=80)
    parser.add_argument("--mf_lr", type=float, default=1e-3)
    parser.add_argument("--save_artifacts", type=int, default=1, help="Save adapter and user embedding artifacts in addition to results.json.")
    parser.add_argument("--init_adapter_path", type=Path, default=None, help="Load a pretrained global adapter as initialization (for cross-domain transfer).")
    parser.add_argument("--save_global_adapter_path", type=Path, default=None, help="Save final global adapter to this path (for use as init in target domain).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    run_start_time = time.perf_counter()
    train_start_time = None
    train_end_time = None
    eval_start_time = None
    eval_end_time = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if not args.data_path.exists():
        alt = Path.cwd() / args.data_path
        if alt.exists():
            args.data_path = alt
    by_user = read_fedcia_dat(args.data_path)
    train, valid, test = leave_one_out_split(by_user)
    if args.train_with_valid_for_test:
        train = {
            user: list(items) + ([valid[user]] if user in valid else [])
            for user, items in train.items()
        }
        args.skip_valid_eval = True
    users = sorted(train)
    max_user = max(users)
    max_item = max(max(items) for items in train.values())
    max_item = max(max_item, max(valid.values()), max(test.values()))
    args.num_users = max_user + 1
    args.observed_num_users = len(users)
    args.num_items = max_item + 1
    print(f"users={len(users)} num_users={args.num_users} num_items={args.num_items}")
    apply_fedbasa_policy(args)
    if args.fedbasa_requested:
        print(
            f"{args.paper_method} policy:",
            {
                "dataset": args.dataset_name,
                "resolved_method": args.fedbasa_resolved_method,
                "adapter_mode": args.fedbasa_adapter_mode,
                "server_mode": args.fedbasa_server_mode,
                "anchor_mode": args.fedbasa_semantic_anchor_mode,
            },
        )
    args.effective_semantic_anchor_weight = resolve_semantic_anchor_weight(args)
    if getattr(args, "prox_auto", 0):
        density = args.observed_num_users / max(1, args.num_items)
        gate = max(0.0, min(1.0, (density - args.semantic_anchor_auto_low) /
                            max(1e-6, args.semantic_anchor_auto_high - args.semantic_anchor_auto_low)))
        args.prox_mu = args.prox_mu * gate
        print("prox_auto:", {"density": density, "gate": gate, "prox_eff": args.prox_mu})
    if args.semantic_anchor_auto > 0:
        print(
            "semantic_anchor_auto:",
            {
                "density": args.observed_num_users / max(1, args.num_items),
                "target": args.semantic_anchor_weight,
                "effective": args.effective_semantic_anchor_weight,
            },
        )

    if args.precomputed_text_features is not None and args.precomputed_text_features.exists():
        text_features = load_precomputed_text_features(args.precomputed_text_features)
        if text_features.size(0) < args.num_items:
            raise ValueError(
                f"precomputed_text_features has {text_features.size(0)} rows, but data needs {args.num_items} items"
            )
        if text_features.size(0) > args.num_items:
            text_features = text_features[:args.num_items]
    else:
        if args.item_titles_json is not None:
            titles = load_titles_json(args.item_titles_json, max_item, args.dataset_name)
        elif args.allow_hash_text_fallback:
            print(f"[OFFLINE MODE] Skipping network download; using hash-based text features.")
            titles = fallback_titles(max_item, args.dataset_name)
        elif args.title_source == "ml100k":
            titles = ensure_ml100k_titles(args.output_dir / "metadata", max_item)
        elif args.title_source == "ml1m":
            titles = ensure_ml1m_titles(args.output_dir / "metadata", max_item)
        else:
            titles = fallback_titles(max_item, args.dataset_name)
        title_tag = args.item_titles_json.stem if args.item_titles_json is not None else args.title_source
        feature_cache = args.output_dir / f"item_text_features_{title_tag}_{args.text_model.replace('/', '_')}.pt"
        text_features = build_text_features(
            titles,
            args.text_model,
            feature_cache,
            args.encode_batch_size,
            device,
            args.allow_hash_text_fallback,
        )
    clients = make_clients(users, args.num_clients, args.seed, train, args.client_split, args.client_dirichlet_alpha)
    clients = [c for c in clients if c]
    client_sizes = [len(client_users) for client_users in clients]
    print(
        "client_split:",
        {
            "mode": args.client_split,
            "alpha": args.client_dirichlet_alpha,
            "num_clients": len(clients),
            "min_size": min(client_sizes) if client_sizes else 0,
            "max_size": max(client_sizes) if client_sizes else 0,
        },
    )

    if args.only_mf_baseline:
        train_start_time = time.perf_counter()
        mf_metrics = train_eval_mf_baseline(train, valid, test, clients, args, device)
        train_end_time = time.perf_counter()
        runtime = {
            "total_seconds": time.perf_counter() - run_start_time,
            "train_seconds": train_end_time - train_start_time,
            "eval_seconds": 0.0,
            "device": str(device),
        }
        if device.type == "cuda":
            runtime["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        result = {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "history": [],
            "runtime": runtime,
            "metrics_valid": None,
            "metrics_no_pap": None,
            "metrics_valid_ditto": None,
            "metrics_ditto": None,
            "metrics_valid_pap": None,
            "metrics_pap": None,
            "metrics_fedmf": mf_metrics,
        }
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("MF/FCF baseline:", mf_metrics)
        print(f"Saved results to {args.output_dir / 'results.json'}")
        return

    global_adapter = create_adapter(args, text_features.size(1), device)
    if args.init_adapter_path is not None and not args.init_adapter_path.exists():
        raise FileNotFoundError(
            f"Pretrained adapter not found: {args.init_adapter_path}. "
            "Cross-domain transfer cannot silently fall back to random initialization."
        )
    if args.init_adapter_path is not None and args.init_adapter_path.exists():
        pretrained_state = torch.load(args.init_adapter_path, map_location="cpu")
        current_keys = set(global_adapter.state_dict().keys())
        pretrained_keys = set(pretrained_state.keys())
        if current_keys == pretrained_keys:
            global_adapter.load_state_dict(pretrained_state)
        else:
            # Architecture mismatch (e.g. Adapter -> FedBATAdapter/MoEAdapter):
            # initialize every branch/expert from the shared source adapter.
            mapped = {}
            prefixes = ["warm.", "expert0.", "expert1.", "global_branch.", "personal_branch."]
            for k, v in pretrained_state.items():
                if k in current_keys:
                    mapped[k] = v
                else:
                    for prefix in prefixes:
                        if prefix + k in current_keys:
                            mapped[prefix + k] = v
            missing = current_keys - set(mapped)
            global_adapter.load_state_dict(mapped, strict=False)
            print(f"Partial adapter load: mapped {len(mapped)} keys, {len(missing)} keys kept random (cold/condition/gate).")
        print(f"Loaded pretrained adapter from {args.init_adapter_path}")
    client_adapters = [
        {k: v.detach().cpu().clone() for k, v in global_adapter.state_dict().items()}
        for _ in clients
    ]
    l2sp_reference = None
    if args.l2sp_mu > 0:
        l2sp_source = args.l2sp_path if args.l2sp_path is not None else args.init_adapter_path
        if l2sp_source is not None and l2sp_source.exists():
            raw_l2sp = torch.load(l2sp_source, map_location="cpu")
            current_keys = set(global_adapter.state_dict().keys())
            mapped = {}
            for key, value in raw_l2sp.items():
                if key in current_keys:
                    mapped[key] = value.detach().to(device)
                else:
                    warm_key = f"warm.{key}"
                    if warm_key in current_keys:
                        mapped[warm_key] = value.detach().to(device)
            l2sp_reference = mapped
            print(f"L2-SP reference loaded: {len(mapped)} matched parameters from {l2sp_source}")
        else:
            print("L2-SP requested but no reference adapter was found; continuing without L2-SP regularization.")
    client_profiles = [client_text_profile(client_users, train, text_features) for client_users in clients]
    coldness_table = item_coldness(train, args.num_items)
    if args.fedbat_routing == "constant":
        coldness_table = torch.full_like(coldness_table, float(coldness_table.mean()))
        print(f"routing=constant: coldness := mean ({float(coldness_table.mean()):.4f})")
    elif args.fedbat_routing == "random":
        # Permute coldness scores across items with a fixed generator so the
        # marginal distribution is preserved but no longer tied to popularity.
        perm_gen = torch.Generator().manual_seed(12345)
        coldness_table = coldness_table[torch.randperm(args.num_items, generator=perm_gen)]
        print("routing=random: coldness scores permuted across items")
    popularity_counts = item_counts(train, args.num_items)
    client_popularity = client_popularity_tables(clients, train, args.num_items)
    user_embeddings = nn.Embedding(args.num_users, text_features.size(1)).to(device)
    nn.init.normal_(user_embeddings.weight, std=0.02)
    init_user_state = {k: v.detach().cpu().clone() for k, v in user_embeddings.state_dict().items()}
    local_user_states = [
        {k: v.detach().cpu().clone() for k, v in init_user_state.items()}
        for _ in clients
    ]
    client_lambda_states = None
    if getattr(args, "fedbat_learn_lambda", 0) and args.method == "fedbat":
        client_lambda_states = [float(args.fedbat_condition_scale) for _ in clients]
    global_item_bias_state = None
    client_item_bias_states = None
    if args.item_bias > 0:
        item_bias = nn.Embedding(args.num_items, 1).to(device)
        nn.init.zeros_(item_bias.weight)
        global_item_bias_state = {k: v.detach().cpu().clone() for k, v in item_bias.state_dict().items()}
        client_item_bias_states = [
            {k: v.detach().cpu().clone() for k, v in global_item_bias_state.items()}
            for _ in clients
        ]
    global_item_residual_state = None
    client_item_residual_states = None
    if args.item_residual > 0:
        item_residual = nn.Embedding(args.num_items, text_features.size(1)).to(device)
        nn.init.zeros_(item_residual.weight)
        global_item_residual_state = {k: v.detach().cpu().clone() for k, v in item_residual.state_dict().items()}
        client_item_residual_states = [
            {k: v.detach().cpu().clone() for k, v in global_item_residual_state.items()}
            for _ in clients
        ]
    server_opt_state = init_server_optimizer_state(global_adapter.state_dict())

    history = []
    swa_global_states = []
    swa_client_states = []
    swa_user_states = []
    train_start_time = time.perf_counter()
    for r in range(args.rounds):
        current_alpha = round_fedbat_alpha(args, r) if args.method == "fedbat" else None
        if current_alpha is not None:
            set_adapter_alpha(global_adapter, current_alpha)
        client_states = []
        user_states = []
        bias_states = []
        residual_states = []
        losses = []
        rate = float(getattr(args, "client_participation", 1.0))
        if rate < 1.0:
            part_rng = random.Random(args.seed * 7919 + r)
            k = max(1, int(round(rate * len(clients))))
            active_cids = sorted(part_rng.sample(range(len(clients)), k))
        else:
            active_cids = list(range(len(clients)))
        for cid in active_cids:
            client_users = clients[cid]
            c_state, u_state, b_state, r_state, lambda_out, loss = train_client(
                client_adapters[cid], local_user_states[cid], text_features, client_users, train, args, device, cid, r,
                client_profiles[cid].to(device), coldness_table, l2sp_reference,
                None if client_item_bias_states is None else client_item_bias_states[cid],
                None if client_item_residual_states is None else client_item_residual_states[cid],
                None if client_lambda_states is None else client_lambda_states[cid],
            )
            client_states.append(c_state)
            user_states.append(u_state)
            if lambda_out is not None:
                client_lambda_states[cid] = lambda_out
            if b_state is not None:
                bias_states.append(b_state)
            if r_state is not None:
                residual_states.append(r_state)
            losses.append(loss)
            if device.type == "cuda":
                del c_state, u_state
                torch.cuda.empty_cache()
        client_states = perturb_client_states(
            client_states, global_adapter.state_dict(), getattr(args, "update_noise", 0.0),
            seed=int(args.seed) * 1000003 + int(r),
        )
        client_states = privatize_client_states(client_states, global_adapter.state_dict(), args, r)
        fedavg_state = aggregate_states(client_states)
        server_state, server_opt_state = server_optimizer_step(
            global_adapter.state_dict(), fedavg_state, server_opt_state, args
        )
        global_adapter.load_state_dict(server_state)
        previous_adapters = client_adapters
        if args.method == "dual" and args.dual_personal_aggregation == "local" and args.aggregation == "fedavg":
            client_adapters = dual_global_local_aggregate_states(client_states)
        elif args.aggregation == "semantic":
            client_adapters = personalized_aggregate_states(
                client_states, client_profiles, args.semantic_temperature, args.semantic_self_weight
            )
        elif args.aggregation == "adaptive":
            client_adapters = adaptive_aggregate_states(
                client_states,
                previous_adapters,
                client_profiles,
                args.semantic_temperature,
                args.semantic_self_weight,
                args.adaptive_update_weight,
            )
        elif args.aggregation == "directional":
            client_adapters = adaptive_aggregate_states(
                client_states,
                previous_adapters,
                client_profiles,
                args.semantic_temperature,
                args.semantic_self_weight,
                1.0,
            )
        elif args.aggregation == "decoupled":
            client_adapters = decoupled_aggregate_states(
                client_states,
                previous_adapters,
                client_profiles,
                args.semantic_temperature,
                args.semantic_self_weight,
                args.decoupled_profile_weight,
                args.decoupled_magnitude_weight,
            )
        elif args.aggregation == "local":
            client_adapters = [
                {k: v.detach().cpu().clone() for k, v in state.items()}
                for state in client_states
            ]
        else:
            client_adapters = [
                {k: v.detach().cpu().clone() for k, v in server_state.items()}
                for _ in clients
            ]
        if args.item_bias > 0 and bias_states:
            global_item_bias_state = aggregate_states(bias_states)
            if args.aggregation == "local":
                client_item_bias_states = [
                    {k: v.detach().cpu().clone() for k, v in state.items()}
                    for state in bias_states
                ]
            else:
                client_item_bias_states = [
                    {k: v.detach().cpu().clone() for k, v in global_item_bias_state.items()}
                    for _ in clients
                ]
        if args.item_residual > 0 and residual_states:
            global_item_residual_state = aggregate_states(residual_states)
            if args.aggregation == "local":
                client_item_residual_states = [
                    {k: v.detach().cpu().clone() for k, v in state.items()}
                    for state in residual_states
                ]
            else:
                client_item_residual_states = [
                    {k: v.detach().cpu().clone() for k, v in global_item_residual_state.items()}
                    for _ in clients
                ]
        if float(getattr(args, "client_participation", 1.0)) < 1.0:
            # Partial participation: merge active clients' states into the full lists.
            new_user_states = list(local_user_states)
            for cid, u_state in zip(active_cids, user_states):
                new_user_states[cid] = u_state
            local_user_states = new_user_states
            if client_item_bias_states is not None and bias_states:
                merged_bias = list(client_item_bias_states)
                for cid, b_state in zip(active_cids, bias_states):
                    merged_bias[cid] = b_state
                bias_states = merged_bias
            if client_item_residual_states is not None and residual_states:
                merged_res = list(client_item_residual_states)
                for cid, r_state in zip(active_cids, residual_states):
                    merged_res[cid] = r_state
                residual_states = merged_res
        else:
            local_user_states = user_states
        if args.swa_tail_rounds > 0 and (r + 1) > args.rounds - args.swa_tail_rounds:
            swa_global_states.append({k: v.detach().cpu().clone() for k, v in global_adapter.state_dict().items()})
            swa_client_states.append([
                {k: v.detach().cpu().clone() for k, v in state.items()}
                for state in client_adapters
            ])
            swa_user_states.append([
                {k: v.detach().cpu().clone() for k, v in state.items()}
                for state in local_user_states
            ])
        row = {"round": r + 1, "train_loss": float(np.mean(losses))}
        if current_alpha is not None:
            row["fedbat_alpha_effective"] = float(current_alpha)
        print(row)
        history.append(row)
    train_end_time = time.perf_counter()

    if args.swa_tail_rounds > 0 and swa_global_states:
        averaged_global = aggregate_states(swa_global_states)
        global_adapter.load_state_dict(averaged_global)
        if swa_client_states:
            averaged_clients = []
            for cid in range(len(clients)):
                averaged_clients.append(aggregate_states([round_states[cid] for round_states in swa_client_states]))
            client_adapters = averaged_clients
        if swa_user_states:
            averaged_users = []
            for cid in range(len(clients)):
                averaged_users.append(aggregate_states([round_states[cid] for round_states in swa_user_states]))
            local_user_states = averaged_users
        history.append({"round": "swa", "tail_rounds": int(args.swa_tail_rounds)})

    use_client_adapters = args.aggregation in {"semantic", "adaptive", "directional", "decoupled", "local"} or (
        args.method == "dual" and args.dual_personal_aggregation == "local"
    )
    eval_adapter_states = client_adapters if use_client_adapters else None
    eval_item_bias_states = client_item_bias_states if args.item_bias > 0 and args.aggregation == "local" else None
    eval_item_residual_states = client_item_residual_states if args.item_residual > 0 and args.aggregation == "local" else None
    eval_start_time = time.perf_counter()
    base_metrics = evaluate(
        global_adapter, local_user_states, None, text_features, clients, train, valid, test, args, device,
        eval_adapter_states, client_profiles, coldness_table, popularity_counts, client_popularity,
        global_item_bias_state, eval_item_bias_states, global_item_residual_state, eval_item_residual_states,
        client_lambdas=client_lambda_states
    )
    valid_metrics = None
    if not args.skip_valid_eval:
        valid_metrics = evaluate(
            global_adapter, local_user_states, None, text_features, clients, train, valid, test, args, device,
            eval_adapter_states, client_profiles, coldness_table, popularity_counts, client_popularity,
            global_item_bias_state, eval_item_bias_states, global_item_residual_state, eval_item_residual_states,
            target_split=valid, client_lambdas=client_lambda_states
        )
    print(f"{args.method} no PAP:", base_metrics)
    if valid_metrics is not None:
        print(f"{args.method} valid no PAP:", valid_metrics)
    ditto_metrics = None
    if args.ditto_steps > 0:
        ditto_states, ditto_user_states, ditto_loss = ditto_personalize(
            global_adapter.state_dict(),
            local_user_states,
            text_features,
            clients,
            train,
            args,
            device,
            client_profiles,
            coldness_table,
        )
        ditto_metrics = evaluate(
            global_adapter, ditto_user_states, None, text_features, clients, train, valid, test, args, device,
            ditto_states, client_profiles, coldness_table, popularity_counts, client_popularity,
            global_item_bias_state, eval_item_bias_states, global_item_residual_state, eval_item_residual_states,
            client_lambdas=client_lambda_states
        )
        ditto_valid_metrics = evaluate(
            global_adapter, ditto_user_states, None, text_features, clients, train, valid, test, args, device,
            ditto_states, client_profiles, coldness_table, popularity_counts, client_popularity,
            global_item_bias_state, eval_item_bias_states, global_item_residual_state, eval_item_residual_states,
            target_split=valid, client_lambdas=client_lambda_states
        )
        ditto_metrics["personalization_loss"] = ditto_loss
        ditto_valid_metrics["personalization_loss"] = ditto_loss
        print(f"{args.method} Ditto-style personalized:", ditto_metrics)
        print(f"{args.method} valid Ditto-style personalized:", ditto_valid_metrics)
    else:
        ditto_valid_metrics = None
    pap_metrics = None
    if args.head_steps > 0:
        heads = train_local_heads(
            global_adapter, local_user_states, text_features, clients, train, args, device,
            eval_adapter_states, client_profiles, coldness_table
        )
        pap_metrics = evaluate(
            global_adapter, local_user_states, heads, text_features, clients, train, valid, test, args, device,
            eval_adapter_states, client_profiles, coldness_table, popularity_counts, client_popularity,
            global_item_bias_state, eval_item_bias_states, global_item_residual_state, eval_item_residual_states,
            client_lambdas=client_lambda_states
        )
        pap_valid_metrics = evaluate(
            global_adapter, local_user_states, heads, text_features, clients, train, valid, test, args, device,
            eval_adapter_states, client_profiles, coldness_table, popularity_counts, client_popularity,
            global_item_bias_state, eval_item_bias_states, global_item_residual_state, eval_item_residual_states,
            target_split=valid, client_lambdas=client_lambda_states
        )
        print(f"{args.method} PAP:", pap_metrics)
        print(f"{args.method} valid PAP:", pap_valid_metrics)
    else:
        pap_valid_metrics = None
    mf_metrics = None
    if args.run_mf_baseline:
        mf_metrics = train_eval_mf_baseline(train, valid, test, clients, args, device)
        print("FedMF baseline:", mf_metrics)
    eval_end_time = time.perf_counter()

    runtime = {
        "total_seconds": time.perf_counter() - run_start_time,
        "train_seconds": None if train_start_time is None or train_end_time is None else train_end_time - train_start_time,
        "eval_seconds": None if eval_start_time is None or eval_end_time is None else eval_end_time - eval_start_time,
        "device": str(device),
    }
    if device.type == "cuda":
        runtime["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "learned_lambda_per_client": [round(float(x), 4) for x in client_lambda_states] if client_lambda_states else None,
        "history": history,
        "runtime": runtime,
        "metrics_valid": valid_metrics,
        "metrics_no_pap": base_metrics,
        "metrics_valid_ditto": ditto_valid_metrics,
        "metrics_ditto": ditto_metrics,
        "metrics_valid_pap": pap_valid_metrics,
        "metrics_pap": pap_metrics,
        "metrics_fedmf": mf_metrics,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_artifacts:
        torch.save(global_adapter.state_dict(), args.output_dir / "global_adapter.pt")
        torch.save(client_adapters, args.output_dir / "client_adapters.pt")
        torch.save(local_user_states, args.output_dir / "local_user_embeddings.pt")
        if args.item_bias > 0:
            torch.save(global_item_bias_state, args.output_dir / "global_item_bias.pt")
            torch.save(client_item_bias_states, args.output_dir / "client_item_bias.pt")
        if args.item_residual > 0:
            torch.save(global_item_residual_state, args.output_dir / "global_item_residual.pt")
            torch.save(client_item_residual_states, args.output_dir / "client_item_residual.pt")
    if args.save_global_adapter_path is not None:
        args.save_global_adapter_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_adapter.state_dict(), args.save_global_adapter_path)
        print(f"Saved global adapter to {args.save_global_adapter_path}")
    print(f"Saved results to {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
