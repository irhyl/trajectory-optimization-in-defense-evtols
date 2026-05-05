# EvTOL-Traj — Overleaf Submission Package

This folder is ready for direct upload to Overleaf.

## Contents

```
submission/
├── neurips_2026.tex      ← Main paper (NeurIPS 2026 E&D track)
├── references.bib        ← BibTeX references
├── neurips_2026.sty      ← NeurIPS style file (download from neurips.cc if missing)
└── figures/              ← 40 PNG figures referenced in the paper
    ├── perception_*.png  ← Perception layer (terrain, wind, SAM threat, fused cost)
    ├── planning_*.png    ← Planning layer (spatial density, Pareto front, cost distributions)
    ├── vehicle_*.png     ← Vehicle layer (BEMT, energy/battery, signatures)
    └── control_*.png     ← Control layer (tracking, PID, mode transitions)
```

## Overleaf Upload Steps

1. Create a new Overleaf project (blank)
2. Upload this entire folder (zip it first: right-click → Send to → Compressed folder)
3. Overleaf will detect `neurips_2026.tex` as the main file
4. Download `neurips_2026.sty` from https://neurips.cc and upload it to the project root
5. Compile with pdflatex

## Figure inventory (40 figures)

| Prefix | Layer | Count | Description |
|--------|-------|-------|-------------|
| `perception_A*` | Perception | 5 | Terrain, wind, obstacles, SAM threat, fused cost |
| `perception_F*` | Perception | 2 | SAM range rings, detection probability curves |
| `perception_H*` | Perception | 1 | Summary dashboard |
| `planning_A*`   | Planning   | 2 | Spatial density, trajectory vectors |
| `planning_B*`   | Planning   | 1 | Path metrics by risk |
| `planning_C*`   | Planning   | 1 | Cost distributions |
| `planning_E*`   | Planning   | 2 | Threat distributions, threat vs risk |
| `planning_F*`   | Planning   | 1 | Feasibility pie charts |
| `planning_H*`   | Planning   | 1 | Correlation matrix |
| `planning_K*`   | Planning   | 2 | Pareto front, cost breakdown pie |
| `vehicle_A*`    | Vehicle    | 1 | Mission profile overview |
| `vehicle_B*`    | Vehicle    | 1 | BEMT rotor performance |
| `vehicle_C*`    | Vehicle    | 1 | Cruise aerodynamics |
| `vehicle_D*`    | Vehicle    | 3 | Energy/battery, planned vs simulated, SOC |
| `vehicle_F*`    | Vehicle    | 1 | Acoustic signatures |
| `vehicle_G*`    | Vehicle    | 1 | Infrared signatures |
| `vehicle_H*`    | Vehicle    | 1 | RCS signatures |
| `vehicle_I*`    | Vehicle    | 1 | Correlation matrix |
| `vehicle_K*`    | Vehicle    | 2 | Multi-signature Pareto 3D, performance radar |
| `control_A*`    | Control    | 2 | Tracking performance, tracking vs mission time |
| `control_B*`    | Control    | 1 | Control effort distributions |
| `control_C*`    | Control    | 1 | Mode transitions |
| `control_D*`    | Control    | 1 | Motor allocation / PWM |
| `control_E*`    | Control    | 1 | Performance by risk |
| `control_F*`    | Control    | 1 | ITAE metrics |
| `control_G*`    | Control    | 1 | Mission success / abort |
| `control_H*`    | Control    | 1 | Cross-correlations |
| `control_Z*`    | Control    | 1 | KPI summary |

## Neural Network Results (Table 3 in paper)

| Region | T1 AUC | T2 R² | T3 R² | T4 R² |
|--------|--------|-------|-------|-------|
| Delhi | 0.537±0.013 | **0.960±0.011** | -1.75 | -0.872 |
| Mumbai | 0.617±0.046 | **0.915±0.008** | -1.53 | -0.809 |
| Bangalore | 0.481±0.014 | **0.898±0.039** | -1.61 | -0.850 |
| Arunachal | 0.809±0.057 | **0.952±0.010** | -1.80 | -0.796 |
| Odisha | 0.463±0.028 | **0.801±0.119** | -2.10 | -0.844 |
| Ladakh | 0.676±0.060 | **0.928±0.011** | -2.89 | -0.729 |
