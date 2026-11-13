"""Score a saved CNN on a subset it was not trained on.

Default: subset1. Pass the cutoff file from the training run so the
yes/no line is not retuned on this exam.

    python scripts/eval_subset.py --checkpoint models/checkpoints/tiny_cnn_subset0_subset2.pt --cutoff-file models/checkpoints/cutoff_subset0_subset2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import TinyNoduleCNN  # noqa: E402
from src.train import PatchDataset, collect_patches, evaluate, patch_cache_tag  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--subset", default="subset1")
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--cutoff-file", default=None)
    parser.add_argument("--checkpoint", default="models/checkpoints/tiny_cnn.pt")
    args = parser.parse_args()

    if args.cutoff is None:
        cutoff_file = (
            Path(args.cutoff_file)
            if args.cutoff_file
            else ROOT / "models" / "checkpoints" / "cutoff.json"
        )
        if not cutoff_file.is_absolute():
            cutoff_file = ROOT / cutoff_file
        if cutoff_file.exists():
            args.cutoff = float(json.loads(cutoff_file.read_text(encoding="utf-8"))["cutoff"])
            print(f"Using frozen cutoff {args.cutoff:.3f} from {cutoff_file}")
        else:
            args.cutoff = 0.32
            print("No cutoff file found; using 0.32")

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["train"]["subset"] = args.subset
    cfg["train"].pop("subsets", None)

    cache = ROOT / cfg["paths"]["processed_dir"] / f"{args.subset}_patches_{patch_cache_tag(cfg)}.npz"
    if cache.exists():
        data = np.load(cache)
        x, y = data["x"], data["y"]
        print(f"Loaded cached patches from {cache}")
    else:
        x, y, _uids = collect_patches(cfg, args.subset)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=x, y=y)
        print(f"Saved patch cache to {cache}")

    print(f"Patches: {len(y)}  nodules: {int(y.sum())}  not-nodules: {int((1 - y).sum())}")
    print("This set was not used to train the CNN. Cutoff is frozen from training (not re-tuned here).")

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    model = TinyNoduleCNN(
        in_channels=cfg["patch"]["context_slices"],
        size_xy=cfg["patch"]["size_xy"],
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    loader = DataLoader(PatchDataset(x, y), batch_size=cfg["train"]["batch_size"])
    scores, gt = evaluate(model, loader)

    print(f"ROC-AUC: {roc_auc_score(gt, scores):.3f}")
    print(f"PR-AUC:  {average_precision_score(gt, scores):.3f}")
    print(f"--- frozen cutoff {args.cutoff:.3f} (from held-out training scans, not retuned) ---")
    print(
        classification_report(
            gt,
            (scores >= args.cutoff).astype(int),
            target_names=["not nodule", "nodule"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
