# Perturbation Size h Calibration

This report records raw data only. It does not select a best h or apply a numerical pass/fail threshold.

## Vanilla determinism (five requested repeats)
- repeat 0: scene_valid=True, EEF_t+1 XYZ=[-0.1549053815822409, -0.009149533136668728, 0.24408066731683326]
- repeat 1: scene_valid=True, EEF_t+1 XYZ=[-0.1549053815822409, -0.009149533136668728, 0.24408066731683326]
- repeat 2: scene_valid=True, EEF_t+1 XYZ=[-0.1549053815822409, -0.009149533136668728, 0.24408066731683326]
- repeat 3: scene_valid=True, EEF_t+1 XYZ=[-0.1549053815822409, -0.009149533136668728, 0.24408066731683326]
- repeat 4: scene_valid=True, EEF_t+1 XYZ=[-0.1549053815822409, -0.009149533136668728, 0.24408066731683326]

## Per-h raw results
### 0.01 m (1.0 cm)
- +x: EEF_t+1=[-0.1549053815822409, -0.009149533136668728, 0.24408066731683326], R=[0.0, 0.0, 0.0], one-sided J_cal=[0.0, 0.0, 0.0]
- -x: EEF_t+1=[-0.1549053815822409, -0.009149533136668728, 0.24408066731683326], R=[0.0, 0.0, 0.0], one-sided J_cal=[-0.0, -0.0, -0.0]
- +y: unavailable (not collected)
- -y: unavailable (not collected)
- central J_cal x: [0.0, 0.0, 0.0]; y: None; 3x2: None
### 0.02 m (2.0 cm)
- +x: unavailable (not collected)
- -x: unavailable (not collected)
- +y: unavailable (not collected)
- -y: unavailable (not collected)
- central J_cal x: None; y: None; 3x2: None
### 0.04 m (4.0 cm)
- +x: unavailable (not collected)
- -x: unavailable (not collected)
- +y: unavailable (not collected)
- -y: unavailable (not collected)
- central J_cal x: None; y: None; 3x2: None
### 0.08 m (8.0 cm)
- +x: unavailable (not collected)
- -x: unavailable (not collected)
- +y: unavailable (not collected)
- -y: unavailable (not collected)
- central J_cal x: None; y: None; 3x2: None
### 0.16 m (16.0 cm)
- +x: unavailable (not collected)
- -x: unavailable (not collected)
- +y: unavailable (not collected)
- -y: unavailable (not collected)
- central J_cal x: None; y: None; 3x2: None
### 0.32 m (32.0 cm)
- +x: unavailable (not collected)
- -x: unavailable (not collected)
- +y: unavailable (not collected)
- -y: unavailable (not collected)
- central J_cal x: None; y: None; 3x2: None

## Raw diagnostics

- `calibration_raw_records.json` and `calibration_conditions.csv`: per-condition requested/actual target poses, validity diagnostics, EEF transition, and actions.
- `j_cal_by_h.json`: one-sided and central J_cal values (3x2 where both axes are available).
- `local_linearity_metrics.json`: scaling residuals only for 1→2, 2→4, 4→8, 8→16, and 16→32 cm; directional raw slope differences; and adjacent-h raw central-column differences.

Unavailable values mean a required condition was scene-invalid or absent; they are never replaced with zero.
