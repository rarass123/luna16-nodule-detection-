"""Open one LUNA16 scan and draw the first known nodule.

Run from the project folder, with the venv active:
    python scripts/explore_one_scan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io import load_mhd, world_to_voxel, window_hu  # noqa: E402


def main() -> None:
    raw = ROOT / "data" / "raw"
    subset = raw / "subset0"
    ann_path = raw / "annotations.csv"
    out_path = ROOT / "notebooks" / "first_nodule.png"

    if not ann_path.exists():
        raise FileNotFoundError(f"Missing {ann_path}")

    mhd_files = sorted(subset.glob("*.mhd"))
    if not mhd_files:
        raise FileNotFoundError(f"No .mhd files in {subset}")

    ann = pd.read_csv(ann_path)
    print("scans found:", len(mhd_files))
    print(ann.head())

    uids = {p.stem for p in mhd_files}
    nodules = ann[ann["seriesuid"].isin(uids)]
    if nodules.empty:
        raise RuntimeError("No annotated nodules in subset0.")

    row = nodules.iloc[0]
    uid = row["seriesuid"]
    scan_path = subset / f"{uid}.mhd"
    volume, image = load_mhd(scan_path)
    print("uid:", uid)
    print("volume shape (z, y, x):", volume.shape)
    print("spacing (x, y, z mm):", image.GetSpacing())
    print("origin:", image.GetOrigin())

    xyz = (row["coordX"], row["coordY"], row["coordZ"])
    vx, vy, vz = world_to_voxel(image, xyz)
    print("world mm:", xyz)
    print("voxel x,y,z:", vx, vy, vz, "diameter_mm:", row["diameter_mm"])

    sl = window_hu(volume[vz])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sl, cmap="gray")
    ax.scatter([vx], [vy], s=80, facecolors="none", edgecolors="lime", linewidths=2)
    ax.set_title(f"{uid[:24]}…  z={vz}")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
