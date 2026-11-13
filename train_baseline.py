"""False-positive reduction baseline (no PyTorch).

Takes candidate points from candidates_V2.csv, crops a small patch around
each one, and trains a logistic regression: nodule vs not-nodule.

Run from the project folder with the venv active:
    python scripts/train_baseline.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io import load_mhd  # noqa: E402
from src.data.patches import extract_candidate_patch  # noqa: E402

CFG = {
    "raw_dir": ROOT / "data" / "raw",
    "candidates": ROOT / "data" / "raw" / "candidates_V2.csv",
    "subset": "subset0",
    "max_scans": None,  # all scans in subset0
    "positives_per_scan": 50,  # take every nodule candidate we can
    "negatives_per_scan": 20,
    "size_xy": 64,
    "context_slices": 3,
    "seed": 42,
    "out_dir": ROOT / "notebooks" / "baseline_examples",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def collect_patches() -> tuple[np.ndarray, np.ndarray]:
    subset_dir = CFG["raw_dir"] / CFG["subset"]
    candidates = pd.read_csv(CFG["candidates"])
    mhd_files = sorted(subset_dir.glob("*.mhd"))
    if CFG["max_scans"] is not None:
        mhd_files = mhd_files[: CFG["max_scans"]]
    if not mhd_files:
        raise FileNotFoundError(f"No .mhd files in {subset_dir}")
    if not CFG["candidates"].exists():
        raise FileNotFoundError(f"Missing {CFG['candidates']}")

    patches, labels = [], []
    for mhd in tqdm(mhd_files, desc="Extracting patches"):
        volume, image = load_mhd(mhd)
        rows = candidates[candidates["seriesuid"] == mhd.stem]
        pos = rows[rows["class"] == 1]
        neg = rows[rows["class"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        sampled = pd.concat(
            [
                pos.sample(n=min(CFG["positives_per_scan"], len(pos)), random_state=CFG["seed"]),
                neg.sample(n=min(CFG["negatives_per_scan"], len(neg)), random_state=CFG["seed"]),
            ]
        )
        for _, row in sampled.iterrows():
            patch = extract_candidate_patch(
                volume,
                image,
                (row["coordX"], row["coordY"], row["coordZ"]),
                size_xy=CFG["size_xy"],
                context_slices=CFG["context_slices"],
            )
            if patch is None:
                continue
            patches.append(patch)
            labels.append(int(row["class"]))

    if not patches:
        raise RuntimeError("No patches extracted. Check candidates_V2.csv matches subset0.")
    return np.stack(patches), np.array(labels)


def save_examples(x: np.ndarray, y: np.ndarray) -> None:
    out = CFG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    mid = CFG["context_slices"] // 2
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for row_i, label in enumerate((1, 0)):
        idxs = np.where(y == label)[0][:4]
        for col_i, ax in enumerate(axes[row_i]):
            ax.axis("off")
            if col_i >= len(idxs):
                continue
            ax.imshow(x[idxs[col_i], mid], cmap="gray")
            ax.set_title("nodule" if label == 1 else "not a nodule")
    fig.tight_layout()
    path = out / "patch_examples.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")


def main() -> None:
    set_seed(CFG["seed"])
    x, y = collect_patches()
    print(f"Patches: {len(y)}  nodules: {int(y.sum())}  not-nodules: {int((1 - y).sum())}")
    save_examples(x, y)

    flat = x.reshape(len(x), -1)
    x_train, x_val, y_train, y_val = train_test_split(
        flat, y, test_size=0.2, random_state=CFG["seed"], stratify=y
    )
    print(f"train: {len(y_train)} (nodules {int(y_train.sum())})  val: {len(y_val)} (nodules {int(y_val.sum())})")

    # balanced = punish ignoring the rare nodule class
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_val)[:, 1]
    pred = (scores >= 0.5).astype(int)

    print(f"val ROC-AUC: {roc_auc_score(y_val, scores):.3f}")
    print(f"val PR-AUC:  {average_precision_score(y_val, scores):.3f}")
    print(
        classification_report(
            y_val,
            pred,
            target_names=["not nodule", "nodule"],
            zero_division=0,
        )
    )
    print(
        "Watch nodule recall (did we catch true nodules?) and PR-AUC, not accuracy. "
        "This is still a baseline, not a hospital system."
    )


if __name__ == "__main__":
    main()
