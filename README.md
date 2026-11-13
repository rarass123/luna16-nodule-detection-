# Lung-nodule candidate classification (LUNA16)

This repository scores official LUNA16 nodule candidates. It does not search a full CT volume, and it does not diagnose cancer.

A nine-slice network reads a 64×64 patch around each candidate in `candidates_V2.csv`. Training uses subset0 and subset2. Scans are split so one CT is never in both training and validation. Subset1 is scored later with the cutoff left frozen.

## Result

On held-out scans inside subset0 and subset2, the precision-recall AUC was 0.84.

On untouched subset1 (170 nodules among 1,389 patches, positive rate about 0.12), the frozen cutoff gave:

- precision-recall AUC 0.78
- ROC AUC 0.91
- precision 0.90
- sensitivity 0.59

The negative patches are a sample of about 20 per scan, not every false mark in the volume. The official LUNA16 detection metric is FROC. This repository does not report FROC.

An earlier version averaged each patch and used a heavier class weight. Its precision-recall AUC was 0.16, near the positive rate of the patches. Both changes were made in the same revision.

The cutoff from that run is in `models/checkpoints/cutoff_subset0_subset2_c9.json`. The network weights are not in this repository.

## Train and score

LUNA16 CT files are not included. Download them from the [LUNA16 site](https://luna16.grand-challenge.org/) into `data/raw/` as `subset0`, `subset1`, and `subset2`, with `annotations.csv` and `candidates_V2.csv`.

```powershell
python -m venv .venv-cnn
.\.venv-cnn\Scripts\Activate.ps1
pip install -r requirements-train.txt
python -m src.train --config configs/default.json
python scripts/eval_subset.py --checkpoint models/checkpoints/tiny_cnn_subset0_subset2_c9.pt --cutoff-file models/checkpoints/cutoff_subset0_subset2_c9.json --subset subset1
```

Training expects the Python environment that can install PyTorch. `requirements.txt` is the lighter set used for exploration.
