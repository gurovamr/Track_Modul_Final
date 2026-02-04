#!/usr/bin/env python3
"""
FirstBlood Numerical Stability Validator

Robust, schema-agnostic assessment of simulation numerical correctness.
- No physiological assumptions
- No unit conversions
- No vessel name hardcoding
- Focuses on: NaN/Inf, blow-up, monotone time, periodic convergence

Outputs: numerical_stability_report.txt + diagnostic plots

Usage:
    python3 validation.py <results_dir> [output_dir]
    
Example:
    python3 validation.py /path/to/results/patient_025 /path/to/output
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import sys
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# STEP 0: SCHEMA INSPECTION & FILE LOADING
# ============================================================================

def inspect_output_schema(results_dir):
    """
    Inspect output folder structure and representative files.
    Returns: (schema_summary, selected_files)
    """
    results_path = Path(results_dir)
    schema_summary = {}
    
    # Check main subfolders
    arterial_dir = results_path / "arterial"
    heart_dir = results_path / "heart_kim_lit"
    
    schema_summary['has_arterial'] = arterial_dir.exists()
    schema_summary['has_heart'] = heart_dir.exists()
    
    selected_files = {
        'arterial': [],
        'heart': [],
    }
    
    # Select representative arterial files
    if arterial_dir.exists():
        vessel_files = list(arterial_dir.glob("*.txt"))
        schema_summary['arterial_total_files'] = len(vessel_files)
        
        # Pick first 2 files
        for i, f in enumerate(sorted(vessel_files)[:2]):
            selected_files['arterial'].append({
                'name': f.name,
                'path': f,
                'type': 'vessel'
            })
    
    # Select representative heart files
    if heart_dir.exists():
        heart_files = list(heart_dir.glob("*.txt"))
        schema_summary['heart_total_files'] = len(heart_files)
        
        # Pick aorta and one chamber file
        for name in ['aorta.txt', 'p_LV1.txt']:
            fpath = heart_dir / name
            if fpath.exists():
                selected_files['heart'].append({
                    'name': fpath.name,
                    'path': fpath,
                    'type': 'heart'
                })
    
    return schema_summary, selected_files


def load_csv_safe(filepath, max_rows=None):
    """Load CSV file robustly, inferring delimiter and handling errors."""
    try:
        data = np.genfromtxt(filepath, delimiter=',', max_rows=max_rows)
        
        # Handle single-line data
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        return data
    except Exception as e:
        print(f"    ⚠️ Error loading {filepath.name}: {e}")
        return None


def analyze_file_schema(filepath):
    """
    Analyze single file: columns, shape, basic stats.
    Returns: {ncols, nrows, col_stats}
    """
    data = load_csv_safe(filepath)
    if data is None:
        return None
    
    nrows, ncols = data.shape
    
    # Per-column statistics
    col_stats = []
    for i in range(ncols):
        col = data[:, i]
        col_stats.append({
            'col_idx': i,
            'min': float(np.min(col)),
            'max': float(np.max(col)),
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'has_nan': bool(np.any(np.isnan(col))),
            'has_inf': bool(np.any(np.isinf(col))),
        })
    
    # Check if col0 is monotone increasing time
    col0_monotone = np.all(np.diff(data[:, 0]) >= 0)
    
    return {
        'nrows': nrows,
        'ncols': ncols,
        'col_stats': col_stats,
        'col0_monotone': col0_monotone,
        'first_3_rows': data[:3, :].tolist() if nrows >= 3 else data.tolist(),
    }


def print_schema_summary(results_dir, selected_files):
    """Print schema inspection to console and file."""
    schema_info = inspect_output_schema(results_dir)
    schema_summary, files = schema_info
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("STEP 0: OUTPUT SCHEMA INSPECTION")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append(f"Results directory: {results_dir}")
    report_lines.append(f"Has arterial folder: {schema_summary['has_arterial']}")
    report_lines.append(f"Has heart folder: {schema_summary['has_heart']}")
    
    if schema_summary['has_arterial']:
        report_lines.append(f"Arterial files: {schema_summary['arterial_total_files']}")
    if schema_summary['has_heart']:
        report_lines.append(f"Heart files: {schema_summary['heart_total_files']}")
    
    report_lines.append("")
    report_lines.append("REPRESENTATIVE FILES ANALYZED:")
    report_lines.append("-" * 80)
    
    all_analyzed = []
    
    # Analyze arterial files
    for f in files['arterial']:
        report_lines.append(f"\n{f['name']}:")
        schema = analyze_file_schema(f['path'])
        if schema:
            report_lines.append(f"  Shape: {schema['nrows']} rows × {schema['ncols']} columns")
            report_lines.append(f"  Col0 monotone increasing (time): {schema['col0_monotone']}")
            report_lines.append(f"  First 3 rows:")
            for row in schema['first_3_rows']:
                report_lines.append(f"    {row}")
            
            report_lines.append(f"  Per-column stats:")
            for stat in schema['col_stats']:
                has_issues = "⚠️" if (stat['has_nan'] or stat['has_inf']) else "✓"
                report_lines.append(f"    Col {stat['col_idx']:2d}: [{stat['min']:12.3e}, {stat['max']:12.3e}]  " +
                                  f"mean={stat['mean']:12.3e}  {has_issues}")
            
            all_analyzed.append((f['name'], schema))
    
    # Analyze heart files
    for f in files['heart']:
        report_lines.append(f"\n{f['name']}:")
        schema = analyze_file_schema(f['path'])
        if schema:
            report_lines.append(f"  Shape: {schema['nrows']} rows × {schema['ncols']} columns")
            report_lines.append(f"  Col0 monotone increasing (time): {schema['col0_monotone']}")
            report_lines.append(f"  First 3 rows:")
            for row in schema['first_3_rows']:
                report_lines.append(f"    {row}")
            
            report_lines.append(f"  Per-column stats:")
            for stat in schema['col_stats']:
                has_issues = "⚠️" if (stat['has_nan'] or stat['has_inf']) else "✓"
                report_lines.append(f"    Col {stat['col_idx']:2d}: [{stat['min']:12.3e}, {stat['max']:12.3e}]  " +
                                  f"mean={stat['mean']:12.3e}  {has_issues}")
            
            all_analyzed.append((f['name'], schema))
    
    report_lines.append("\n" + "=" * 80)
    
    return "\n".join(report_lines), all_analyzed


# ============================================================================
# STEP 1: GLOBAL STABILITY CHECKS
# ============================================================================

def select_primary_signals(results_dir):
    """
    Select 3-5 diverse signals from different locations.
    Returns list of (filepath, signal_name, col_idx).
    """
    arterial_dir = Path(results_dir) / "arterial"
    heart_dir = Path(results_dir) / "heart_kim_lit"
    
    signals = []
    
    # From arterial: pick first, middle, last by vessel ID
    if arterial_dir.exists():
        vessel_files = sorted([f for f in arterial_dir.glob("*.txt")])
        
        if len(vessel_files) > 0:
            # First vessel (proximal)
            signals.append((vessel_files[0], f"{vessel_files[0].stem}_proximal", 1))
            
            # Middle vessel
            if len(vessel_files) > 2:
                signals.append((vessel_files[len(vessel_files)//2], 
                              f"{vessel_files[len(vessel_files)//2].stem}_mid", 1))
            
            # Last vessel (distal)
            if len(vessel_files) > 1:
                signals.append((vessel_files[-1], f"{vessel_files[-1].stem}_distal", 1))
    
    # From heart: pick aorta and one chamber
    if heart_dir.exists():
        aorta_file = heart_dir / "aorta.txt"
        if aorta_file.exists():
            signals.append((aorta_file, "heart_aorta", 1))
        
        p_lv = heart_dir / "p_LV1.txt"
        if p_lv.exists():
            signals.append((p_lv, "heart_lv_pressure", 1))
    
    return signals[:5]  # Return up to 5 signals


def check_global_stability(results_dir):
    """
    STEP 1: Global stability checks - NaN/Inf, blow-up, time consistency.
    Returns: report_lines, check_results
    """
    report_lines = []
    report_lines.append("\n" + "=" * 80)
    report_lines.append("STEP 1: GLOBAL STABILITY CHECKS")
    report_lines.append("=" * 80)
    
    signals = select_primary_signals(results_dir)
    check_results = {}
    
    for filepath, signal_name, col_idx in signals:
        report_lines.append(f"\n{signal_name}:")
        
        data = load_csv_safe(filepath)
        if data is None:
            report_lines.append("  ❌ Failed to load")
            continue
        
        nrows, ncols = data.shape
        
        # Check col_idx is valid
        if col_idx >= ncols:
            col_idx = 1  # Default to column 1 if out of range
        
        time_col = data[:, 0]
        signal_col = data[:, col_idx]
        
        # 1. NaN/Inf check
        has_nan = np.any(np.isnan(signal_col))
        has_inf = np.any(np.isinf(signal_col))
        
        if has_nan or has_inf:
            report_lines.append(f"  ❌ NaN/Inf detected: NaN={has_nan}, Inf={has_inf}")
            check_results[signal_name] = 'FAIL'
            continue
        
        # 2. Blow-up check
        signal_max = np.max(np.abs(signal_col))
        blowup_detected = signal_max > 1e10
        
        if blowup_detected:
            report_lines.append(f"  ❌ Blow-up detected: max(|signal|) = {signal_max:.3e}")
            check_results[signal_name] = 'FAIL'
            continue
        
        # 3. Time monotonicity
        time_monotone = np.all(np.diff(time_col) >= 0)
        time_monotone_strict = np.all(np.diff(time_col) > 0)
        
        if not time_monotone:
            report_lines.append(f"  ❌ Time not monotone increasing")
            check_results[signal_name] = 'FAIL'
            continue
        
        # 4. Time step consistency
        dt = np.diff(time_col)
        dt_mean = np.mean(dt)
        dt_std = np.std(dt)
        dt_min = np.min(dt)
        dt_max = np.max(dt)
        dt_irregular = dt_std / (dt_mean + 1e-16) > 0.5  # >50% variation
        
        report_lines.append(f"  ✓ Numerically stable (no NaN/Inf/blow-up)")
        report_lines.append(f"    Time range: [{time_col[0]:.6e}, {time_col[-1]:.6e}] s")
        report_lines.append(f"    Signal range: [{np.min(signal_col):.6e}, {np.max(signal_col):.6e}]")
        report_lines.append(f"    Δt: mean={dt_mean:.6e} s, std={dt_std:.6e} s")
        report_lines.append(f"    Δt range: [{dt_min:.6e}, {dt_max:.6e}] s")
        
        if dt_irregular:
            report_lines.append(f"    ⚠️ Δt has >50% variation (irregular time stepping)")
            check_results[signal_name] = 'WARN'
        else:
            report_lines.append(f"    ✓ Δt consistent (variation < 50%)")
            check_results[signal_name] = 'PASS'
    
    report_lines.append("\n" + "-" * 80)
    report_lines.append("Summary:")
    for sig, status in check_results.items():
        report_lines.append(f"  {sig}: {status}")
    
    return "\n".join(report_lines), check_results


# ============================================================================
# STEP 2: PERIODIC CONVERGENCE CHECK
# ============================================================================

def estimate_period_autocorr(signal_data, time_data, window_frac=0.3):
    """
    Estimate cardiac period using autocorrelation on late-time window.
    Returns: (period, confidence_peak_prominence)
    """
    # Use late 30% of signal for more stable oscillations
    start_idx = int(len(signal_data) * (1 - window_frac))
    late_signal = signal_data[start_idx:]
    late_time = time_data[start_idx:]
    
    # Detrend and normalize
    late_signal = signal.detrend(late_signal)
    late_signal = (late_signal - np.mean(late_signal)) / (np.std(late_signal) + 1e-16)
    
    # Autocorrelation
    acf = np.correlate(late_signal, late_signal, mode='full')
    acf = acf[len(acf)//2:]  # Keep positive lags only
    acf = acf / acf[0]  # Normalize
    
    # Find peaks in autocorrelation
    # Skip first few samples (autocorr at lag 0 is always 1)
    min_lag_samples = max(5, int(0.01 * len(late_signal)))
    acf_for_peaks = acf[min_lag_samples:]
    
    peaks, properties = signal.find_peaks(acf_for_peaks, height=0.3, distance=5)
    
    if len(peaks) == 0:
        return None, None
    
    # Use first prominent peak
    best_peak_idx = peaks[np.argmax(properties['peak_heights'])]
    lag_samples = best_peak_idx + min_lag_samples
    
    # Convert lag in samples to time
    mean_dt = np.mean(np.diff(late_time))
    period = lag_samples * mean_dt
    prominence = properties['peak_heights'][np.argmax(properties['peak_heights'])]
    
    return period, prominence


def extract_cycles(signal_data, time_data, period, n_cycles=3):
    """
    Extract last N cycles from signal given estimated period.
    Returns: list of (start_idx, end_idx, period_actual)
    """
    cycles = []
    
    # Find last N cycle boundaries
    t_end = time_data[-1]
    
    for i in range(n_cycles):
        end_time = t_end - i * period
        start_time = end_time - period
        
        if start_time < time_data[0]:
            continue
        
        # Find indices
        end_idx = np.searchsorted(time_data, end_time)
        start_idx = np.searchsorted(time_data, start_time)
        
        if start_idx >= end_idx or start_idx < 0:
            continue
        
        actual_period = time_data[end_idx-1] - time_data[start_idx]
        cycles.append((start_idx, end_idx, actual_period))
    
    return list(reversed(cycles))  # Return in chronological order


def compute_rms_percent_error(cycle1, cycle2):
    """
    RMS% = RMS(cycle1 - cycle2) / mean(|cycle2|) * 100
    """
    # Interpolate to same length
    n = max(len(cycle1), len(cycle2))
    t1 = np.linspace(0, 1, len(cycle1))
    t2 = np.linspace(0, 1, len(cycle2))
    t_common = np.linspace(0, 1, n)
    
    c1_interp = np.interp(t_common, t1, cycle1)
    c2_interp = np.interp(t_common, t2, cycle2)
    
    mse = np.mean((c1_interp - c2_interp)**2)
    denominator = np.mean(np.abs(c2_interp)) + 1e-16
    
    rms_pct = 100 * np.sqrt(mse) / denominator
    
    return rms_pct


def check_periodic_convergence(results_dir):
    """
    STEP 2: Periodic convergence check.
    Returns: report_lines, convergence_results, all_cycles_data
    """
    report_lines = []
    report_lines.append("\n" + "=" * 80)
    report_lines.append("STEP 2: PERIODIC CONVERGENCE CHECK")
    report_lines.append("=" * 80)
    
    signals = select_primary_signals(results_dir)
    convergence_results = {}
    all_cycles_data = {}  # For plotting
    
    for filepath, signal_name, col_idx in signals:
        report_lines.append(f"\n{signal_name}:")
        
        data = load_csv_safe(filepath)
        if data is None:
            report_lines.append("  ❌ Failed to load")
            continue
        
        nrows, ncols = data.shape
        if col_idx >= ncols:
            col_idx = 1
        
        time_col = data[:, 0]
        signal_col = data[:, col_idx]
        
        # Estimate period
        period, prominence = estimate_period_autocorr(signal_col, time_col)
        
        if period is None:
            report_lines.append("  ⚠️ Could not estimate period from autocorrelation")
            convergence_results[signal_name] = {
                'status': 'INSUFFICIENT_DATA',
                'period': None,
                'cycles': 0,
            }
            continue
        
        report_lines.append(f"  Estimated period: {period:.6e} s (prominence={prominence:.3f})")
        
        # Extract cycles
        cycles = extract_cycles(signal_col, time_col, period, n_cycles=5)
        
        if len(cycles) < 2:
            report_lines.append(f"  ⚠️ Only {len(cycles)} cycles detected")
            convergence_results[signal_name] = {
                'status': 'INSUFFICIENT_CYCLES',
                'period': float(period),
                'cycles': len(cycles),
            }
            continue
        
        report_lines.append(f"  ✓ Extracted {len(cycles)} cycles")
        
        # Compute convergence
        cycle_signals = []
        for start_idx, end_idx, _ in cycles:
            cycle_signals.append(signal_col[start_idx:end_idx])
        
        all_cycles_data[signal_name] = {
            'time': time_col,
            'signal': signal_col,
            'cycles': cycles,
            'cycle_signals': cycle_signals,
            'period': period,
        }
        
        # RMS error between consecutive cycles
        rms_errors = []
        for i in range(len(cycle_signals) - 1):
            rms_pct = compute_rms_percent_error(cycle_signals[i], cycle_signals[i+1])
            rms_errors.append(rms_pct)
        
        rms_errors = np.array(rms_errors)
        
        report_lines.append(f"  Cycle-to-cycle RMS% errors:")
        for i, rms_err in enumerate(rms_errors):
            report_lines.append(f"    Cycle {i} → {i+1}: {rms_err:.4f}%")
        
        # Determine pass/fail/warn
        mean_rms = np.mean(rms_errors)
        rms_decreasing = np.all(np.diff(rms_errors) <= 0)
        
        if mean_rms < 0.1:
            status = 'PASS'
        elif mean_rms < 1.0 or rms_decreasing:
            status = 'WARN'
        else:
            status = 'FAIL'
        
        report_lines.append(f"  Mean RMS%: {mean_rms:.4f}% → Status: {status}")
        
        convergence_results[signal_name] = {
            'status': status,
            'period': float(period),
            'cycles': len(cycles),
            'rms_errors': rms_errors.tolist(),
            'mean_rms_pct': float(mean_rms),
            'rms_decreasing': bool(rms_decreasing),
        }
    
    report_lines.append("\n" + "-" * 80)
    
    return "\n".join(report_lines), convergence_results, all_cycles_data


# ============================================================================
# STEP 3: JUNCTION MASS CONSERVATION (optional)
# ============================================================================

def check_mass_conservation(results_dir):
    """
    STEP 3: Check mass conservation at junctions (if files exist).
    For now: placeholder noting that junction files need explicit topology map.
    """
    report_lines = []
    report_lines.append("\n" + "=" * 80)
    report_lines.append("STEP 3: JUNCTION MASS CONSERVATION")
    report_lines.append("=" * 80)
    report_lines.append("\nNote: Junction mass conservation requires topology map.")
    report_lines.append("This is deferred until topology is explicitly provided.")
    report_lines.append("Mass conservation can be verified by summing flows at")
    report_lines.append("known branch points (if topology documented).")
    
    return "\n".join(report_lines)


# ============================================================================
# STEP 4: PLOTTING & VISUALIZATION
# ============================================================================

def plot_signal_cycles(output_dir, all_cycles_data):
    """Plot overlay of last 3-5 cycles for each signal."""
    output_dir = Path(output_dir)
    
    for signal_name, data in all_cycles_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cycles = data['cycles']
        time_col = data['time']
        cycle_signals = data['cycle_signals']
        
        # Plot each cycle
        colors = plt.cm.viridis(np.linspace(0, 1, len(cycle_signals)))
        
        for i, (start_idx, end_idx, _) in enumerate(cycles):
            t_cycle = time_col[start_idx:end_idx]
            t_normalized = (t_cycle - t_cycle[0]) / (t_cycle[-1] - t_cycle[0])
            
            ax.plot(t_normalized, cycle_signals[i], 'o-', markersize=2, 
                   label=f'Cycle {i}', color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Normalized time within cycle')
        ax.set_ylabel('Signal value')
        ax.set_title(f'Cycle Overlay: {signal_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        safe_name = signal_name.replace('/', '_')
        plt.savefig(output_dir / f"signal_overlay_{safe_name}.png", dpi=100, bbox_inches='tight')
        plt.close()


def plot_rms_convergence(output_dir, convergence_results):
    """Plot RMS% error per cycle pair for each signal."""
    output_dir = Path(output_dir)
    
    for signal_name, result in convergence_results.items():
        if 'rms_errors' not in result or len(result['rms_errors']) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rms_errors = result['rms_errors']
        cycle_pairs = np.arange(1, len(rms_errors) + 1)
        
        ax.plot(cycle_pairs, rms_errors, 'o-', markersize=8, linewidth=2, color='darkblue')
        ax.axhline(y=0.1, color='green', linestyle='--', linewidth=2, label='Target: 0.1%')
        ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='Warning: 1.0%')
        
        ax.set_xlabel('Cycle pair')
        ax.set_ylabel('RMS% error')
        ax.set_title(f'Convergence: {signal_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        safe_name = signal_name.replace('/', '_')
        plt.savefig(output_dir / f"rms_convergence_{safe_name}.png", dpi=100, bbox_inches='tight')
        plt.close()


def plot_dt_histogram(output_dir, results_dir):
    """Plot time step distribution."""
    signals = select_primary_signals(results_dir)
    
    all_dts = []
    for filepath, _, _ in signals:
        data = load_csv_safe(filepath)
        if data is not None:
            dt = np.diff(data[:, 0])
            all_dts.extend(dt)
    
    if not all_dts:
        return
    
    all_dts = np.array(all_dts)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(all_dts, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Time step Δt (s)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Time Step Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Time series of Δt
    ax2.plot(all_dts, linewidth=0.5, color='darkblue')
    ax2.set_xlabel('Time step index')
    ax2.set_ylabel('Δt (s)')
    ax2.set_title('Time Step Variation')
    ax2.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "dt_histogram.png", dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def run_validation(results_dir, output_dir=None):
    """
    Execute complete numerical stability validation.
    """
    results_dir = Path(results_dir)
    
    if output_dir is None:
        output_dir = Path.cwd() / "validation_output_numerical"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"FIRSTBLOOD NUMERICAL STABILITY VALIDATION")
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # STEP 0: Schema inspection
    # ========================================================================
    print("STEP 0: Inspecting output schema...")
    schema_report, analyzed_files = print_schema_summary(str(results_dir), None)
    print(schema_report)
    
    # ========================================================================
    # STEP 1: Global stability
    # ========================================================================
    print("\nSTEP 1: Checking global stability...")
    stability_report, stability_checks = check_global_stability(str(results_dir))
    print(stability_report)
    
    # ========================================================================
    # STEP 2: Periodic convergence
    # ========================================================================
    print("\nSTEP 2: Checking periodic convergence...")
    convergence_report, convergence_results, all_cycles_data = check_periodic_convergence(str(results_dir))
    print(convergence_report)
    
    # ========================================================================
    # STEP 3: Mass conservation
    # ========================================================================
    print("\nSTEP 3: Checking mass conservation...")
    conservation_report = check_mass_conservation(str(results_dir))
    print(conservation_report)
    
    # ========================================================================
    # Compile full report
    # ========================================================================
    full_report = "\n".join([
        schema_report,
        stability_report,
        convergence_report,
        conservation_report,
    ])
    
    report_file = output_dir / "numerical_stability_report.txt"
    with open(report_file, 'w') as f:
        f.write(full_report)
    
    print(f"\n✓ Report saved: {report_file}")
    
    # ========================================================================
    # Generate plots
    # ========================================================================
    print("\nGenerating diagnostic plots...")
    
    plot_signal_cycles(output_dir, all_cycles_data)
    plot_rms_convergence(output_dir, convergence_results)
    plot_dt_histogram(output_dir, str(results_dir))
    
    print(f"✓ Plots saved to: {output_dir}/")
    
    # ========================================================================
    # Summary table (convergence results)
    # ========================================================================
    convergence_df = pd.DataFrame([
        {
            'signal': name,
            'status': result['status'],
            'period_s': result.get('period', None),
            'cycles_detected': result.get('cycles', 0),
            'mean_rms_pct': result.get('mean_rms_pct', None),
        }
        for name, result in convergence_results.items()
    ])
    
    csv_file = output_dir / "convergence_summary.csv"
    convergence_df.to_csv(csv_file, index=False)
    print(f"✓ Summary table saved: {csv_file}")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs in: {output_dir}/")
    print("  • numerical_stability_report.txt")
    print("  • signal_overlay_*.png")
    print("  • rms_convergence_*.png")
    print("  • dt_histogram.png")
    print("  • convergence_summary.csv")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validation.py <results_dir> [output_dir]")
        print("\nExample:")
        print("  python3 validation.py /path/to/results/patient_025")
        print("  python3 validation.py /path/to/results/patient_025 /path/to/output")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_validation(results_dir, output_dir)
