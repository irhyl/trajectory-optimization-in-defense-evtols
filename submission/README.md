# EvTOL-Traj — Overleaf Submission Package

This folder is ready for direct upload to Overleaf.

## Contents

```text
submission/
├── neurips_2026.tex      ← Main paper (NeurIPS 2026 E&D track)
├── references.bib        ← BibTeX references (14 entries)
├── neurips_2026.sty      ← Download from neurips.cc and place here
└── figures/              ← 40 PNG figures (all cited figures confirmed present)
```

## Overleaf Steps

1. Zip this folder → Upload to Overleaf as new project
2. Download `neurips_2026.sty` from <https://neurips.cc> → upload to project root
3. Set main file to `neurips_2026.tex` → Compile with pdflatex

## Dataset Summary (16 Regions, 400,000 records)

| Region | Setting | Records | Primary SAMs |
| --- | --- | --- | --- |
| Delhi | Urban / Indo-Gangetic plain | 100,000 | S-300V, SA-11, SA-22 |
| Mumbai | Coastal / naval SAM | 20,000 | S-125, Barak-8, Akash |
| Bangalore | Deccan plateau ~920m | 20,000 | Akash Mk2, QRSAM, SA-6 |
| Arunachal | Himalayan border / LAC | 20,000 | HQ-9B, HQ-16, HQ-17A |
| Odisha | Bay of Bengal coast | 20,000 | Barak-8, SA-3, Akash |
| Ladakh | High-alt desert ~3500m | 20,000 | HQ-9B, SA-15, ZU-23 |
| Srinagar | Kashmir Valley / Pakistan LAC | 20,000 | Spyder, SA-15, HQ-7 |
| Chennai | Coromandel coast | 20,000 | Barak-8, S-125, Akash |
| Kolkata | Gangetic delta / Bangladesh border | 20,000 | SA-6, Akash, S-125 |
| Pune | Western Ghats foothills | 20,000 | Akash Mk2, QRSAM, SA-3 |
| Jaisalmer | Thar Desert / Pakistan border | 20,000 | S-300V, SA-15, Spyder |
| Visakhapatnam | Eastern coast / naval base | 20,000 | Barak-8, SA-3, QRSAM |
| Guwahati | Brahmaputra valley / NE corridor | 20,000 | SA-11, Akash, HQ-16 |
| Port Blair | Andaman Islands / maritime SAM | 20,000 | Barak-8, SA-22, Akash |
| Jodhpur | Thar Desert / air base | 20,000 | S-300V, SA-11, Akash Mk2 |
| Imphal | Manipur valley / Myanmar border | 20,000 | SA-15, HQ-17A, Akash |

## Neural Network Results — Original 6 Regions (Table 3 in paper)

Single-task ST-NN, 3-fold CV, leak-free features, weighted BCE (w+ = n-/n+):

| Region | T1 AUC | T1 F1* | T2 R² | T2 MAE (Wh) |
| --- | --- | --- | --- | --- |
| Delhi | 0.998 ± 0.001 | 0.952 ± 0.006 | 0.968 ± 0.014 | 422 ± 93 |
| Mumbai | 0.992 ± 0.003 | 0.884 ± 0.028 | 0.933 ± 0.022 | 636 ± 57 |
| Bangalore | 0.966 ± 0.003 | 0.939 ± 0.006 | 0.917 ± 0.004 | 703 ± 20 |
| Arunachal | --- (2 pos.) | --- | 0.810 ± 0.188 | 1081 ± 646 |
| Odisha | 0.995 ± 0.001 | 0.990 ± 0.002 | 0.888 ± 0.007 | 860 ± 24 |
| Ladakh | 0.690 ± 0.211 | 0.019 ± 0.010 | 0.771 ± 0.220 | 1170 ± 736 |

\* F1 at threshold tuned per validation fold (not fixed 0.5)

### Key findings

- **T1 AUC >= 0.966** on all 4 regions with adequate class representation
- **T2 R² = 0.77–0.97** from planning-layer physics features alone
- **T3/T4**: open challenges requiring sequence-level encodings (see paper §5.3)
- **Ladakh T1 variance**: only 13 positives in 2000 samples (0.65% rate)
- **Arunachal T1**: 2 positives — degenerate, excluded from evaluation

### Bugs fixed during evaluation

1. Weighted BCE gradient sign was inverted → AUC collapsed to ~0.03 → fixed
2. Target columns leaked into feature matrix (energy, threat, alt_error) → removed
3. T3 head used sigmoid (output 0-1) for logit-space target → changed to linear
4. T3/T4 gradient explosion from linear head in MTL → separated into single-task

## New Regions (10 added)

The 10 new regions use the same 4-layer simulation stack (perception → planning → vehicle → control).
Vehicle simulation uses full BEMT + 2-RC battery physics (same as original 6 regions).
Control layer uses a physics-consistent analytical approximation — position/altitude/attitude
error metrics derived from vehicle outputs without running the full 50 Hz PID loop.

To evaluate all 16 regions: `python scripts/ml/eval_singletask.py`

To regenerate all datasets: `python scripts/expand_dataset_10x.py`
