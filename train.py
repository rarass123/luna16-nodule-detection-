"""Train TinyNoduleCNN on candidate patches.

Use the Python 3.14 CNN venv, not the 3.15 explore venv:
    .\\.venv-cnn\\Scripts\\Activate.ps1
    python -m src.train --config configs/default.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.io import load_mhd
from src.data.patches import extract_candidate_patch
from src.models import TinyNoduleCNN


class PatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, augment: bool = False) -> None:
        self.patches = torch.from_numpy(patches)
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                patch = torch.flip(patch, dims=(-1,))
            if torch.rand(1).item() < 0.5:
                patch = torch.flip(patch, dims=(-2,))
        return patch, self.labels[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subset_names(cfg: dict) -> list[str]:
    """Training subsets. subset1 stays out of this list; it is the held-out exam."""
    names = cfg["train"].get("subsets")
    if names:
        return [str(name) for name in names]
    return [str(cfg["train"]["subset"])]


def collect_patches(cfg: dict, subset: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    candidates = pd.read_csv(cfg["paths"]["candidates"])
    subset_name = subset or str(cfg["train"]["subset"])
    subset_dir = raw_dir / subset_name
    mhd_files = sorted(subset_dir.glob("*.mhd"))
    max_scans = cfg["train"].get("max_scans")
    if max_scans is not None:
        mhd_files = mhd_files[: int(max_scans)]
    if not mhd_files:
        raise FileNotFoundError(
            f"No .mhd files in {subset_dir}. Unzip {subset_name}.zip there. See docs/LUNA16.md."
        )

    pos_n = cfg["train"]["positives_per_scan"]
    neg_n = cfg["train"]["negatives_per_scan"]
    patches, labels, uids = [], [], []

    for mhd in tqdm(mhd_files, desc="Extracting patches"):
        volume, image = load_mhd(mhd)
        rows = candidates[candidates["seriesuid"] == mhd.stem]
        pos = rows[rows["class"] == 1]
        neg = rows[rows["class"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        sampled = pd.concat(
            [
                pos.sample(n=min(pos_n, len(pos)), random_state=cfg["train"]["seed"]),
                neg.sample(n=min(neg_n, len(neg)), random_state=cfg["train"]["seed"]),
            ]
        )
        for _, row in sampled.iterrows():
            patch = extract_candidate_patch(
                volume,
                image,
                (row["coordX"], row["coordY"], row["coordZ"]),
                size_xy=cfg["patch"]["size_xy"],
                context_slices=cfg["patch"]["context_slices"],
                hu_min=cfg["patch"]["hu_min"],
                hu_max=cfg["patch"]["hu_max"],
            )
            if patch is None:
                continue
            patches.append(patch)
            labels.append(int(row["class"]))
            uids.append(mhd.stem)

    if not patches:
        raise RuntimeError("No patches extracted. Check that candidates_V2.csv matches the subset.")
    return np.stack(patches), np.array(labels), np.array(uids)


def patch_cache_tag(cfg: dict) -> str:
    """Keep 3-slice caches separate from 9-slice caches."""
    return f"c{cfg['patch']['context_slices']}_s{cfg['patch']['size_xy']}"


def load_or_collect_one(cfg: dict, subset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = Path(cfg["paths"]["processed_dir"]) / f"{subset}_patches_{patch_cache_tag(cfg)}_scan.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        print(f"Loaded cached patches from {cache}")
        return data["x"], data["y"], data["uids"]
    x, y, uids = collect_patches(cfg, subset)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y, uids=uids)
    print(f"Saved patch cache to {cache}")
    return x, y, uids


def load_or_collect_patches(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = subset_names(cfg)
    parts = [load_or_collect_one(cfg, name) for name in names]
    print("Training subsets: " + ", ".join(names))
    if len(parts) == 1:
        return parts[0]
    x = np.concatenate([part[0] for part in parts])
    y = np.concatenate([part[1] for part in parts])
    uids = np.concatenate([part[2] for part in parts])
    return x, y, uids


def scan_level_split(
    uids: np.ndarray, y: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep every scan in train or val, never both (no leaked slices from the same CT)."""
    unique = np.unique(uids)
    has_pos = np.array([(y[uids == s] == 1).any() for s in unique])
    try:
        train_scans, val_scans = train_test_split(
            unique, test_size=val_fraction, random_state=seed, stratify=has_pos
        )
    except ValueError:
        train_scans, val_scans = train_test_split(
            unique, test_size=val_fraction, random_state=seed
        )
    train_mask = np.isin(uids, train_scans)
    val_mask = np.isin(uids, val_scans)
    print(
        f"scan-level split: {len(train_scans)} train scans, {len(val_scans)} val scans "
        f"(a scan never appears in both)"
    )
    return train_mask, val_mask


def cutoff_with_best_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    """Try many yes/no cutoffs; return the one with highest nodule F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(precision[i]), float(recall[i]), float(f1[i])


def evaluate(model: TinyNoduleCNN, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            prob = torch.sigmoid(model(xb))
            preds.append(prob.numpy())
            gts.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    set_seed(cfg["train"]["seed"])
    x, y, uids = load_or_collect_patches(cfg)
    print(f"Patches: {len(y)}  nodules: {int(y.sum())}  not-nodules: {int((1 - y).sum())}")

    train_mask, val_mask = scan_level_split(
        uids, y, cfg["train"]["val_fraction"], cfg["train"]["seed"]
    )
    print(
        f"train: {int(train_mask.sum())} (nodules {int(y[train_mask].sum())})  "
        f"val: {int(val_mask.sum())} (nodules {int(y[val_mask].sum())})"
    )

    train_loader = DataLoader(
        PatchDataset(x[train_mask], y[train_mask], augment=True),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        PatchDataset(x[val_mask], y[val_mask]),
        batch_size=cfg["train"]["batch_size"],
    )

    n_pos = max(int(y[train_mask].sum()), 1)
    n_neg = max(int(train_mask.sum()) - n_pos, 1)
    # milder than n_neg/n_pos so the CNN does not shout "nodule" at everything
    pos_weight = torch.tensor(float(np.sqrt(n_neg / n_pos)))
    print(f"pos_weight={pos_weight.item():.2f} (lower than last run → calmer detector)")

    model = TinyNoduleCNN(
        in_channels=cfg["patch"]["context_slices"],
        size_xy=cfg["patch"]["size_xy"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_ap = -1.0
    best_state = None
    best_scores = None
    best_gt = None
    best_epoch = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(yb)
        scores, gt = evaluate(model, val_loader)
        auc = roc_auc_score(gt, scores)
        ap = average_precision_score(gt, scores)
        mark = ""
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_scores, best_gt = scores, gt
            best_epoch = epoch + 1
            mark = "  ← best PR-AUC so far"
        print(
            f"epoch {epoch + 1}/{cfg['train']['epochs']}  "
            f"loss={running / int(train_mask.sum()):.4f}  val ROC-AUC={auc:.3f}  val PR-AUC={ap:.3f}{mark}"
        )

    print(f"\nBest epoch: {best_epoch}  PR-AUC={best_ap:.3f}")
    print("--- at cutoff 0.5 (often a bad fit for a calm model) ---")
    print(
        classification_report(
            best_gt,
            (best_scores >= 0.5).astype(int),
            target_names=["not nodule", "nodule"],
            zero_division=0,
        )
    )
    cut, prec, rec, f1 = cutoff_with_best_f1(best_gt, best_scores)
    print(f"--- cutoff with highest F1 on held-out training scans: {cut:.3f} ---")
    print(f"nodule precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print(
        classification_report(
            best_gt,
            (best_scores >= cut).astype(int),
            target_names=["not nodule", "nodule"],
            zero_division=0,
        )
    )

    out = Path("models/checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    names = subset_names(cfg)
    if len(names) == 1:
        ckpt_name = "tiny_cnn.pt"
        cutoff_name = "cutoff.json"
    else:
        tag = "_".join(names) + f"_c{cfg['patch']['context_slices']}"
        ckpt_name = f"tiny_cnn_{tag}.pt"
        cutoff_name = f"cutoff_{tag}.json"
    ckpt_path = out / ckpt_name
    cutoff_path = out / cutoff_name
    torch.save(best_state, ckpt_path)
    cutoff_path.write_text(
        json.dumps(
            {
                "cutoff": cut,
                "best_epoch": best_epoch,
                "val_pr_auc": best_ap,
                "subsets": names,
                "context_slices": cfg["patch"]["context_slices"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {ckpt_path} (best PR-AUC, not last epoch)")
    print(f"Saved {cutoff_path} — freeze this cutoff when scoring subset1")
    print(
        "Then run: python scripts/eval_subset.py "
        f"--checkpoint {ckpt_path.as_posix()} --cutoff-file {cutoff_path.as_posix()}"
    )


if __name__ == "__main__":
    main()
