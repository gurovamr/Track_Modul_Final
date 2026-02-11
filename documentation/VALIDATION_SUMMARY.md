# FirstBlood Pipeline: Numerical Stability Validation

## Overview

A robust, schema-agnostic numerical stability validator for FirstBlood hemodynamic simulations.

**Location**: `/home/maryyds/final/first_blood/pipeline/validation.py`

**Output directory**: `/home/maryyds/final/first_blood/pipeline/output/`

## What Was Built

### The Validator Script
- **File**: `validation.py` (700+ lines)
- **No dependencies on**: Vessel naming, unit systems, physiological assumptions
- **Focuses on**: Numerical correctness, stability, convergence

### Key Features

#### Step 0: Output Schema Inspection
Automatically discovers and analyzes:
- Folder structure (arterial/, heart_kim_lit/)
- File count and structure (rows × columns)
- Data quality (NaN/Inf detection, monotonicity)
- Representative file analysis (first 3 rows, per-column statistics)

#### Step 1: Global Stability Checks
Validates 5 diverse signals (proximal vessel, mid vessel, distal vessel, heart aorta, heart LV):
- **NaN/Inf check**: FAIL if any detected
- **Blow-up detection**: FAIL if max|signal| > 1e10
- **Time monotonicity**: FAIL if non-monotone
- **Time step consistency**: WARN if Δt variation > 50%, PASS if < 50%

#### Step 2: Periodic Convergence
For each signal:
1. **Auto-estimate period** using autocorrelation on late-time window
2. **Extract 3-5 cycles** from end of simulation
3. **Compute RMS% error** between consecutive cycles
4. **Status assignment**:
   - PASS: RMS% < 0.1%
   - WARN: RMS% < 1.0% OR clearly decreasing trend
   - FAIL: RMS% ≥ 1.0% AND not decreasing

#### Step 3: Mass Conservation
Currently deferred (placeholder for topology-based analysis).

### Outputs Generated

Located in `pipeline/output/`:

**Report**:
- `numerical_stability_report.txt` (7.7 KB)
  - Complete validation documentation
  - Per-signal analysis
  - All statistics and diagnostics

**Data Tables**:
- `convergence_summary.csv`
  - Signal name, status, period, cycles detected, mean RMS%

**Visualizations**:
- `signal_overlay_*.png` (5 plots × 80-90 KB)
  - Overlay of last cycles for each signal
  - Shows visual convergence
  
- `rms_convergence_*.png` (5 plots × 20 KB)
  - RMS% error per cycle pair
  - Reference lines: PASS (0.1%), WARN (1.0%)
  
- `dt_histogram.png` (36 KB)
  - Time step distribution
  - Consistency analysis

## Results for Patient_025

### Global Stability: ✓ PASS
All 5 signals numerically stable:
- No NaN/Inf/blow-up detected
- Time monotone and consistent (Δt = 1e-3 s, std ≈ 0)
- 10,318 time steps over 10.32 seconds

### Periodic Convergence: Mixed
| Signal | Period | Cycles | RMS% | Status |
|--------|--------|--------|------|--------|
| A1_proximal | 0.787 s | 5 | 0.173% | WARN |
| A99_mid | 0.788 s | 5 | 0.143% | WARN |
| p9_distal | 0.793 s | 5 | 0.022% | **PASS** ✓ |
| heart_aorta | 0.787 s | 5 | 0.173% | WARN |
| heart_lv_pressure | 0.793 s | 5 | 0.055% | **PASS** ✓ |

**Interpretation**:
- ✓ Distal signals (periphery, LV) show excellent convergence (RMS% < 0.1%)
- ⚠️ Proximal signals (aorta, main arteries) still settling (RMS% ≈ 0.17%)
- ✓ Clear decreasing trend in most proximal errors
- ✓ No oscillations or instabilities

### Conclusion
**Numerically stable and converging to periodic solution.**
Longer simulations would improve proximal convergence.

## Usage

### Basic
```bash
cd /home/maryyds/final/first_blood/pipeline
python3 validation.py \
  /home/maryyds/final/first_blood/projects/simple_run/results/patient_025 \
  ./output
```

### Output
All files saved to `pipeline/output/`:
```
numerical_stability_report.txt     (full report)
convergence_summary.csv           (summary table)
signal_overlay_*.png              (5 plots)
rms_convergence_*.png             (5 plots)
dt_histogram.png                  (time step analysis)
```

## Design Decisions

### No Physiological Assumptions
- Does NOT convert units (works in raw Pa, m³/s, etc.)
- Does NOT check blood pressure ranges
- Does NOT subtract atmospheric pressure
- Does NOT identify specific vessels

**Why**: Allows objective assessment of solver behavior regardless of parameter correctness.

### Robust Schema Handling
- Auto-discovers output structure
- Infers column indices from data properties
- Handles variable-column files gracefully
- Reports failures clearly without crashing

### Cycle Extraction via Autocorrelation
- Does NOT assume cardiac period (e.g., 1.0 s)
- Auto-estimates from signal using autocorrelation
- Works for any signal frequency
- Reports period + confidence metrics

### RMS% as Convergence Metric
- Independent of absolute signal magnitude
- Works for tiny signals (peripheral flow) and large signals (pressure)
- Clear physical interpretation: % change cycle-to-cycle
- Standard in 1D hemodynamics literature

## Integration Points

### With data_generation.py
- Validator is downstream
- Takes output from FirstBlood solver
- No feedback to data generation

### With visualization/plotting
- Generates standalone PNG plots
- No external plotting dependencies
- Compatible with PDF conversion, web viewing, etc.

### With CI/CD (future)
- `convergence_summary.csv` can be parsed by automated tests
- Threshold checks: "status must be PASS or WARN"
- Report generated for every simulation run

## Next Steps

### Optional Enhancements
1. **Add junction analysis** if topology file provided
2. **Compare across patients** (statistical RMS% patterns)
3. **Extended time step analysis** (CFL checking, adaptive stepping)
4. **Energy conservation** checks (if energy fields available)

### Publication Integration
- Include convergence plots in methods
- Report periods and RMS% in supplemental materials
- Use for validation of baseline Abel_ref2 model

---

## File Manifest

```
/home/maryyds/final/first_blood/pipeline/
├── validation.py                      ← Main validator script
├── README_VALIDATION.md               ← User guide
├── VALIDATION_SUMMARY.md              ← This file
└── output/
    ├── numerical_stability_report.txt
    ├── convergence_summary.csv
    ├── signal_overlay_*.png (5 files)
    ├── rms_convergence_*.png (5 files)
    └── dt_histogram.png
```

## Contact & Support

For issues or questions about numerical validation:
- Check `numerical_stability_report.txt` for diagnostic details
- Verify output folder has results before running validator
- Ensure results folder has `arterial/` and `heart_kim_lit/` subfolders
