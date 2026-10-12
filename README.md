# NPIE — Neural Physical-impairment Estimation for RF Fingerprint Identification

**NPIE** is a transmitter RF-fingerprint identification (RFFI) method built for the
hard case where the *receiver* and the *capture time* both shift between training and
test (cross-receiver × cross-day generalization).

Instead of learning a black-box feature that silently absorbs receiver/time nuisance,
NPIE **explicitly estimates** a compact transmitter hardware-impairment vector
&nbsp;θ_Tx ∈ ℝ³²&nbsp; with a small *ParamNet*, and constructs an invariance:

- a **complex, resolution-preserving front-end** that preserves the second-order
  cyclostationary structure real-valued convolutions destroy;
- a **physical branch** (ParamNet) supervised so that θ_Tx is receiver- and
  time-invariant *by construction* (SupCon + a gradient-reversal branch);
- **phase-jitter augmentation** at train time (multiply the complex IQ input by a
  random rotation e^{jφ}, φ∼U(0,2π)) so the representation is rotation-invariant;
- at test time the classifier is fed θ_Tx ⊕ signal-energy features, and the
  physical branch arbitrates under drift.

The released code reproduces the ManySig (WiSig) results reported in the paper.

---

## Repository structure

```
NPIE/
├── npie.py            # core: NPIE model, losses, data loading, CRD/hardday/lorx splits
├── conjugate_scf.py   # conjugate spectral-correlation (SCF) α-profile feature extractor
├── prepare_data.py    # precompute SCF features from raw ManySig (.pkl/.mat) -> .npz
├── train_phase.py     # E1-PHASE: phase-jitter augmentation (main CRD result)
├── train_base.py      # E1: three-protocol generalization probe (crd/hardday/lorx)
├── train_hos.py       # E1-HOS: higher-order-statistics branch variant
├── train_ens.py       # E1-ENS: soft-voting ensemble variant
├── make_fig_arch.py   # generate the NPIE architecture figure (Fig. 1)
├── paper/             # LaTeX source + figures of the manuscript
│   ├── main.tex
│   ├── refs.bib
│   └── figs/
├── docs/
│   └── DATASET.md     # how to obtain / lay out the ManySig dataset
├── requirements.txt
├── LICENSE
└── .gitignore
```

> **Note on the UAV experiments.** The cross-burst / cross-distance UAV results in the
> paper use a separate preprocessing + training pipeline (`train_uav_npie.py`,
> `preprocess_uav_npie.py`, `sanity_uav.py`) that lives on the GPU training server and
> is **not included in this snapshot yet**. It will be added in a follow-up commit.
> The ManySig (WiSig) code above is complete and self-contained.

---

## Installation

```bash
conda create -n npie python=3.10 -y
conda activate npie
pip install -r requirements.txt      # torch, numpy, matplotlib
```

(Tested with PyTorch 2.x + CUDA 11.8/12.x. CPU runs but is slow.)

---

## Dataset preparation

1. Obtain the **WiSig / ManySig** dataset (6 Tx × 12 Rx × 4 days). See
   [`docs/DATASET.md`](docs/DATASET.md) for the exact source and how to lay it out.
2. Precompute the SCF features (one-time, ~minutes on CPU):

```bash
python prepare_data.py --src raw_wisig/ManySig --out data_cache/wisig_manySig_scf.npz
```

This writes `data_cache/wisig_manySig_scf.npz` consumed by every training script.

---

## Training & reproduction

All training scripts read the precomputed `.npz` via `--npz` and accept `--seeds`.

```bash
# 1) Main CRD result (cross-receiver x cross-day, 4-fold Rx CV, day0-1 / day2-3)
CUDA_VISIBLE_DEVICES=4 python train_phase.py --npz data_cache/wisig_manySig_scf.npz --seeds 0,1,2

# 2) Three-protocol generalization probe
CUDA_VISIBLE_DEVICES=4 python train_base.py --protocol crd      --seeds 0,1,2
CUDA_VISIBLE_DEVICES=4 python train_base.py --protocol hardday  --seeds 0,1,2
CUDA_VISIBLE_DEVICES=5 python train_base.py --protocol lorx --lorx_heldout 0,4,8 --seeds 0,1,2

# 3) Variants
CUDA_VISIBLE_DEVICES=4 python train_hos.py --npz data_cache/wisig_manySig_scf.npz --seeds 0,1,2
CUDA_VISIBLE_DEVICES=5 python train_ens.py --npz data_cache/wisig_manySig_scf.npz --seeds 0,1,2
```

Aggregate the printed `[E1]` lines. The 6-seed soft-voting ensemble is produced by
running multiple `--seeds` and averaging softmax outputs (see `train_ens.py`).

### Architecture figure

```bash
python make_fig_arch.py     # -> paper/figs/fig_arch.pdf (+ .png)
```

---

## Reported results (ManySig / WiSig)

| Setting | NPIE (+phase) | Comparator | Gap |
|---|---|---|---|
| CRD (cross-Rx × cross-day), 6-seed | **96.64 %** | WaveMLP 93.13 % (reported) | +3.5 pp |
| hardday (train day0 / test day3) | **97.93 %** | WaveMLP@1000 92.33 % | **+5.60 pp** |
| lorx (leave-one-Rx-out, 12 Rx) | **97.58 %** | base 97.32 % (neutral) | +0.26 pp |
| WaveMLP@1000 (fresh 3-seed mean) | — | 93.82 ± 1.01 % | — |
| DRIFT (arXiv 2510.09405) cross-Rx / cross-day | — | 75.62 % / 79.27 % | NPIE ≳ +17 pp |

θ_Tx probing: centroid 63.5 % (best 90.45 %) — i.e. θ_Tx separates transmitters
and is far tighter intra-class than inter-class.

> Numbers are the values reported in the manuscript; see `paper/main.tex` for the full
> tables, ablations, and per-receiver / per-fold breakdowns.

---

## Citation

```bibtex
@misc{npie2026,
  title  = {Neural Physical-impairment Estimation for Cross-Receiver and Cross-Time
            Robust RF Fingerprint Identification},
  author = {The NPIE Authors},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

(Update with the final venue once accepted.)

---

## License

MIT — see [LICENSE](LICENSE).
