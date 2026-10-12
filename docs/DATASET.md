# Dataset: WiSig / ManySig

The ManySig subset of the **WiSig** WiFi RF-fingerprint dataset is used for all
ManySig (cross-receiver × cross-day) experiments in the paper.

## What it is

- Modality: WiFi (802.11) IQ captures.
- Composition used here: **6 transmitters × 12 receivers × 4 capture days**.
- Per (Tx, Rx, Day) pair: **1000 IQ captures** (native count — *do not subsample to
  100, or the cross-receiver ceiling is artificially capped below the SOTA*).
- Each capture is a complex IQ vector; the released code consumes a 2-channel
  representation `(I, Q)`.

## Obtain the data

> **TODO (author):** confirm and paste the official download link here before pushing.
> ManySig ships with the WiSig release. Place the raw file as:

```
raw_wisig/ManySig/ManySig.pkl      # (or .mat; --fmt auto detects)
```

`prepare_data.py` reads either format (`--fmt auto|pkl|mat`).

## Precompute features (one-time)

```bash
python prepare_data.py --src raw_wisig/ManySig --out data_cache/wisig_manySig_scf.npz
```

- `--samples_per_pair` default 1000 (use the full native count — see note above).
- `--equalized` 0/1: keep or drop the hardware equalizer output (ManySig provides
  both equalized and non-equalized captures).
- `--jobs` parallel workers for the SCF extraction.

Output: `data_cache/wisig_manySig_scf.npz`, an `(N, 2, 51, 23)` array of conjugate
SCF α-profiles plus labels, consumed by every training script via `--npz`.

## Layout expected by the repo

```
NPIE/
├── raw_wisig/ManySig/ManySig.pkl     # <-- you provide this (gitignored)
└── data_cache/wisig_manySig_scf.npz  # <-- generated (gitignored)
```

Both directories are excluded by `.gitignore`; **do not commit the raw data or the
precomputed cache.**
