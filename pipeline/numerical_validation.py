#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import sys
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


@dataclass
class IntegrityCheck:
    filename: str
    has_nan: bool
    has_inf: bool
    time_monotone: bool
    signal_valid: bool
    dt_mean: Optional[float]
    dt_std: Optional[float]
    dt_min: Optional[float]
    dt_max: Optional[float]
    dt_cv: Optional[float]
    signal_std: Optional[float]
    signal_max_abs: Optional[float]


@dataclass
class ConvergenceResult:
    signal_name: str
    status: str
    period_seconds: Optional[float]
    num_cycles: int
    rms_errors_percent: List[float]
    mean_rms_percent: Optional[float]


def list_models(results_base_path: Path) -> List[str]:
    """
    Available model directories in results folder are identified and returned as a sorted list.
    """
    if not results_base_path.exists():
        return []
    
    # Model directories within results folder are collected
    models = [d.name for d in results_base_path.iterdir() 
              if d.is_dir() and (d / 'arterial').exists()]
    return sorted(models)


def select_model_interactive(results_base_path: Path) -> Optional[str]:
    """
    Models are listed and the user is prompted to select one. Returns the selected model name or None if cancelled.
    """
    models = list_models(results_base_path)
    
    if not models:
        print("No model directories found in results folder.")
        return None
    
    print("\nAvailable models:")
    # Model list with numbering is displayed to the user
    for i, model in enumerate(models, 1):
        print("  {}. {}".format(i, model))
    
    while True:
        selection = input("\nSelect model (type name or number): ").strip()
        
        # Numeric selection is checked against the model list
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print("Invalid number. Try again.")
        elif selection in models:
            return selection
        else:
            print("Model not found. Try again.")


def discover_timeseries_files(results_dir: Path, max_depth: int = 4) -> List[Path]:
    """
    All .txt timeseries files are found recursively up to max_depth directories deep and returned as a sorted list.
    """
    txt_files = []
    
    # All .txt files are recursively collected up to the specified depth
    for item in results_dir.rglob('*.txt'):
        depth = len(item.relative_to(results_dir).parts)
        if depth <= max_depth:
            txt_files.append(item)
    
    return sorted(txt_files)


def load_timeseries(filepath: Path) -> Optional[np.ndarray]:
    """
    Timeseries data is loaded from a text file, attempting comma delimiters first then whitespace. Headers and blank lines are handled.
    """
    try:
        # Data is loaded with comma delimiter first
        data = np.genfromtxt(filepath, delimiter=',', max_rows=None)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        return data
    except Exception:
        try:
            # If comma delimiter fails, whitespace delimiter is attempted
            data = np.genfromtxt(filepath, delimiter=None, max_rows=None)
            
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            return data
        except Exception:
            return None


def detect_time_column(data: np.ndarray) -> Optional[int]:
    """
    The column containing strictly increasing time values is identified. Column index is returned, or None if not found.
    """
    if data.ndim != 2 or data.shape[0] < 10:
        return None
    
    # Each column is checked for strictly increasing values
    for col_idx in range(data.shape[1]):
        col = data[:, col_idx]
        
        # NaN and Inf values are skipped
        if np.any(np.isnan(col)) or np.any(np.isinf(col)):
            continue
        
        # Differences between consecutive values are checked
        diffs = np.diff(col)
        
        # Column is validated as monotonically increasing
        if np.all(diffs > 1e-12) and np.mean(diffs) > 0:
            return col_idx
    
    return None


def compute_dt_stats(time_col: np.ndarray) -> Dict[str, float]:
    """
    Statistics on time step sizes are computed including mean, standard deviation, min, max, and coefficient of variation.
    """
    # Time differences are calculated and their statistics are computed
    dt = np.diff(time_col)
    dt_mean = float(np.mean(dt))
    dt_std = float(np.std(dt))
    dt_min = float(np.min(dt))
    dt_max = float(np.max(dt))
    dt_cv = float(dt_std / (dt_mean + 1e-16))
    
    return {
        'dt_mean': dt_mean,
        'dt_std': dt_std,
        'dt_min': dt_min,
        'dt_max': dt_max,
        'dt_cv': dt_cv,
    }


def check_integrity(data: np.ndarray, filepath: Path) -> IntegrityCheck:
    """
    Numerical integrity of loaded data is checked, including validation of NaN/Inf values, time monotonicity, and signal properties.
    """
    nrows, ncols = data.shape
    time_col_idx = detect_time_column(data)
    
    # Data integrity is initialized with conservative defaults
    result = IntegrityCheck(
        filename=filepath.name,
        has_nan=bool(np.any(np.isnan(data))),
        has_inf=bool(np.any(np.isinf(data))),
        time_monotone=False,
        signal_valid=False,
        dt_mean=None,
        dt_std=None,
        dt_min=None,
        dt_max=None,
        dt_cv=None,
        signal_std=None,
        signal_max_abs=None,
    )
    
    # Quick validation checks - data with NaN or Inf is rejected
    if result.has_nan or result.has_inf:
        return result
    
    # Data with insufficient rows or columns is rejected
    if ncols < 2 or nrows < 100:
        return result
    
    # Time column is required for data validation
    if time_col_idx is None:
        return result
    
    time_col = data[:, time_col_idx]
    # Monotonicity of time is verified
    time_monotone = np.all(np.diff(time_col) > 0)
    
    # Non-monotone time series are flagged as invalid
    if not time_monotone:
        return result
    
    result.time_monotone = True
    # Detailed time step statistics are computed
    dt_stats = compute_dt_stats(time_col)
    result.dt_mean = dt_stats['dt_mean']
    result.dt_std = dt_stats['dt_std']
    result.dt_min = dt_stats['dt_min']
    result.dt_max = dt_stats['dt_max']
    result.dt_cv = dt_stats['dt_cv']
    
    # Signal properties are extracted and validated
    signal_col = data[:, 1] if ncols > 1 else data[:, time_col_idx]
    signal_std = float(np.std(signal_col))
    signal_max_abs = float(np.max(np.abs(signal_col)))
    
    # Signal is validated: non-trivial magnitude and reasonable amplitude
    signal_valid = (signal_std > 1e-16 and signal_max_abs < 1e10)
    
    result.signal_valid = signal_valid
    result.signal_std = signal_std
    result.signal_max_abs = signal_max_abs
    
    return result


def estimate_period_autocorr(signal_data: np.ndarray, 
                             time_data: np.ndarray,
                             window_frac: float = 0.3) -> Tuple[Optional[float], Optional[float]]:
    """
    Cardiac period is estimated via autocorrelation on the late-time window. Returns (period_seconds, prominence) or (None, None).
    """
    start_idx = int(len(signal_data) * (1 - window_frac))
    late_signal = signal_data[start_idx:]
    late_time = time_data[start_idx:]
    
    # Signal is detrended and normalized for autocorrelation analysis
    late_signal = signal.detrend(late_signal)
    late_signal = (late_signal - np.mean(late_signal)) / (np.std(late_signal) + 1e-16)
    
    # Autocorrelation is computed from the normalized signal
    acf = np.correlate(late_signal, late_signal, mode='full')
    acf = acf[len(acf)//2:]
    # ACF is normalized by its maximum value
    acf = acf / (acf[0] + 1e-16)
    
    # Minimum lag is set to exclude near-zero lags
    min_lag_samples = max(5, int(0.01 * len(late_signal)))
    acf_for_peaks = acf[min_lag_samples:]
    
    # Peaks are identified in the ACF to find periodic components
    peaks, properties = signal.find_peaks(acf_for_peaks, height=0.3, distance=5)
    
    # No periodicity is found if no significant peaks are detected
    if len(peaks) == 0:
        return None, None
    
    # Strongest peak is identified as the dominant period
    best_peak_idx = peaks[np.argmax(properties['peak_heights'])]
    lag_samples = best_peak_idx + min_lag_samples
    
    # Period is converted from sample lag to time units
    mean_dt = np.mean(np.diff(late_time))
    period = lag_samples * mean_dt
    prominence = float(properties['peak_heights'][np.argmax(properties['peak_heights'])])
    
    return period, prominence


def extract_cycles(signal_data: np.ndarray, 
                   time_data: np.ndarray, 
                   period: float, 
                   n_cycles: int = 3) -> List[Tuple[int, int, float]]:
    """
    The last N cycles are extracted from the signal given an estimated period. Returns a list of (start_idx, end_idx, period_actual).
    """
    cycles = []
    t_end = time_data[-1]
    
    # Last N cycles are extracted backwards from the end of the signals
    for i in range(n_cycles):
        end_time = t_end - i * period
        start_time = end_time - period
        
        # Cycles outside the time domain are skipped
        if start_time < time_data[0]:
            continue
        
        # Time indices are found via binary search
        end_idx = np.searchsorted(time_data, end_time)
        start_idx = np.searchsorted(time_data, start_time)
        
        # Invalid ranges are skipped
        if start_idx >= end_idx or start_idx < 0:
            continue
        
        # Actual period is measured from the data
        actual_period = time_data[end_idx-1] - time_data[start_idx]
        cycles.append((start_idx, end_idx, actual_period))
    
    return list(reversed(cycles))


def compute_cycle_rms(cycle1: np.ndarray, cycle2: np.ndarray) -> float:
    """
    RMS percent error between two cycles is computed using the formula: RMS% = RMS(cycle1 - cycle2) / mean(|cycle2|) * 100
    """
    # Cycles are normalized to the same time domain via interpolation
    n = max(len(cycle1), len(cycle2))
    t1 = np.linspace(0, 1, len(cycle1))
    t2 = np.linspace(0, 1, len(cycle2))
    t_common = np.linspace(0, 1, n)
    
    # Both cycles are resampled to the common time domain
    c1_interp = np.interp(t_common, t1, cycle1)
    c2_interp = np.interp(t_common, t2, cycle2)
    
    # RMS error is calculated
    mse = np.mean((c1_interp - c2_interp)**2)
    denominator = np.mean(np.abs(c2_interp)) + 1e-16
    
    # RMS is normalized by the reference signal magnitude
    rms_pct = 100.0 * np.sqrt(mse) / denominator
    
    return rms_pct


def analyze_convergence(results_dir: Path) -> Tuple[List[ConvergenceResult], Dict[str, Any]]:
    """
    Periodic convergence on key signals is analyzed. Results are returned as a list along with cycle data for plotting.
    """
    # All timeseries files are discovered within the results directory
    files = discover_timeseries_files(results_dir)
    
    # Candidate files are prioritized: aorta, arterial, and heart signals are preferred
    candidate_files = []
    for f in files:
        path_str = str(f).lower()
        if any(x in path_str for x in ['aorta', 'arterial', 'heart']):
            candidate_files.append(f)
    
    if not candidate_files:
        candidate_files = files[:5]
    else:
        candidate_files = candidate_files[:5]
    
    # Analysis results and cycle data for visualization are collected
    convergence_results = []
    all_cycles_data = {}
    
    # Each candidate file is processed for convergence analysis
    for filepath in candidate_files:
        # Timeseries data is loaded and validated
        data = load_timeseries(filepath)
        
        # Files with insufficient data are skipped
        if data is None or data.shape[0] < 100 or data.shape[1] < 2:
            continue
        
        # Time column is identified
        time_col_idx = detect_time_column(data)
        if time_col_idx is None:
            continue
        
        # Time and signal columns are extracted
        time_col = data[:, time_col_idx]
        signal_col = data[:, 1]
        
        # Cardiac period is estimated from autocorrelation
        period, prominence = estimate_period_autocorr(signal_col, time_col)
        
        # Periodicity must be detected to proceed
        if period is None:
            continue
        
        # Cycles are extracted from the signal
        cycles = extract_cycles(signal_col, time_col, period, n_cycles=5)
        
        # Sufficient cycles are required for convergence analysis
        if len(cycles) < 2:
            continue
        
        # Signal data is extracted for each cycle
        cycle_signals = []
        for start_idx, end_idx, _ in cycles:
            cycle_signals.append(signal_col[start_idx:end_idx])
        
        # RMS errors between consecutive cycles are computed
        rms_errors = []
        for i in range(len(cycle_signals) - 1):
            rms_pct = compute_cycle_rms(cycle_signals[i], cycle_signals[i+1])
            rms_errors.append(rms_pct)
        
        rms_errors = np.array(rms_errors)
        mean_rms = float(np.mean(rms_errors))
        
        # Convergence status is determined from mean RMS error
        if mean_rms < 0.1:
            status = 'PASS'
        elif mean_rms < 1.0:
            status = 'WARN'
        else:
            status = 'FAIL'
        
        # Convergence result is stored
        result = ConvergenceResult(
            signal_name=filepath.stem,
            status=status,
            period_seconds=float(period),
            num_cycles=len(cycles),
            rms_errors_percent=rms_errors.tolist(),
            mean_rms_percent=mean_rms,
        )
        
        convergence_results.append(result)
        # Cycle data is stored for later visualization
        all_cycles_data[filepath.stem] = {
            'time': time_col,
            'signal': signal_col,
            'cycles': cycles,
            'cycle_signals': cycle_signals,
            'period': period,
        }
    
    return convergence_results, all_cycles_data


def write_reports(output_dir: Path, 
                  integrity_checks: List[IntegrityCheck],
                  convergence_results: List[ConvergenceResult],
                  model_name: str):
    """
    Text and JSON reports are generated and written to the output directory.
    """
    # Output directory is created
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("=" * 80)
    lines.append("NUMERICAL STABILITY VALIDATION REPORT")
    # Report header with metadata is generated
    lines.append("=" * 80)
    lines.append("")
    lines.append("Model: {}".format(model_name))
    lines.append("Timestamp: {}".format(pd.Timestamp.now()))
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("INTEGRITY CHECKS")
    lines.append("-" * 80)
    
    pass_count = sum(1 for c in integrity_checks if c.signal_valid)
    fail_count = len(integrity_checks) - pass_count
    
    # Integrity check summary is reported
    lines.append("Files analyzed: {}".format(len(integrity_checks)))
    lines.append("Files passed: {}".format(pass_count))
    lines.append("Files failed: {}".format(fail_count))
    lines.append("")
    
    # Details of failed files are reported
    for check in integrity_checks[:20]:
        status_str = "PASS" if check.signal_valid else "FAIL"
        lines.append("[{}] {}".format(status_str, check.filename))
        
        # Specific issues are documented
        if check.has_nan:
            lines.append("      - Contains NaN values")
        if check.has_inf:
            lines.append("      - Contains Inf values")
        if not check.time_monotone:
            lines.append("      - Time not monotone")
        if check.dt_cv is not None and check.dt_cv > 0.5:
            lines.append("      - Time step variation > 50% (CV={:.3f})".format(check.dt_cv))
        if check.signal_max_abs is not None and check.signal_max_abs > 1e9:
            lines.append("      - Signal amplitude very large (max={:.2e})".format(check.signal_max_abs))
    
    lines.append("")
    lines.append("-" * 80)
    lines.append("TIME STEP STATISTICS")
    lines.append("-" * 80)
    
    if integrity_checks:
        valid_checks = [c for c in integrity_checks if c.dt_mean is not None]
        if valid_checks:
            dt_means = [c.dt_mean for c in valid_checks]
            dt_cvs = [c.dt_cv for c in valid_checks]
            
            lines.append("Mean dt across files: {:.6e} s".format(np.mean(dt_means)))
            lines.append("Median dt: {:.6e} s".format(np.median(dt_means)))
            lines.append("Mean CV (variation): {:.3f}".format(np.mean(dt_cvs)))
    
    lines.append("")
    lines.append("-" * 80)
    lines.append("PERIODICITY AND CONVERGENCE")
    lines.append("-" * 80)
    
    # Convergence results are reported
    if convergence_results:
        for conv in convergence_results:
            lines.append("{}: {}".format(conv.signal_name, conv.status))
            lines.append("  Period: {:.6e} s".format(conv.period_seconds))
            # Detailed convergence metrics are listed
            lines.append("  Cycles: {}".format(conv.num_cycles))
            if conv.mean_rms_percent is not None:
                lines.append("  Mean RMS%: {:.4f}".format(conv.mean_rms_percent))
    else:
        lines.append("No convergence data available.")
    
    lines.append("")
    lines.append("=" * 80)
    
    # Overall validation status is determined
    overall_status = "FAIL"
    if fail_count == 0:
        overall_status = "PASS"
    elif fail_count < len(integrity_checks) * 0.1:
        overall_status = "WARN"
    
    if convergence_results:
        if all(r.status == "PASS" for r in convergence_results):
            if overall_status != "FAIL":
                # Status is updated if all convergence results pass
                overall_status = "PASS"
    
    lines.append("FINAL STATUS: {}".format(overall_status))
    lines.append("=" * 80)
    
    text_report = "\n".join(lines)
    
    # Text report is written to file
    report_file = output_dir / "numerical_stability_report.txt"
    with open(report_file, 'w') as f:
        f.write(text_report)
    
    json_data = {
        'model': model_name,
        'timestamp': str(pd.Timestamp.now()),
        'overall_status': overall_status,
        # Summary statistics are included
        'integrity': {
            'total_files': len(integrity_checks),
            'passed_files': pass_count,
            'failed_files': fail_count,
        },
        'convergence': [asdict(r) for r in convergence_results],
    }
    
    # JSON report is written to file
    json_file = output_dir / "numerical_stability_report.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)


def plot_diagnostics(output_dir: Path, all_cycles_data: Dict[str, Any]):
    """
    Diagnostic plots are generated for cycles and convergence behavior.
    """
    # Output directory is prepared for plots
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for signal_name, data in all_cycles_data.items():
        try:
            # Figure is created for cycle overlay visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cycles = data['cycles']
            time_col = data['time']
            cycle_signals = data['cycle_signals']
            
            # Color scheme spans the cycle range
            colors = plt.cm.viridis(np.linspace(0, 1, len(cycle_signals)))
            
            # Each cycle is normalized and plotted for comparison
            for i, (start_idx, end_idx, _) in enumerate(cycles):
                t_cycle = time_col[start_idx:end_idx]
                t_normalized = (t_cycle - t_cycle[0]) / (t_cycle[-1] - t_cycle[0] + 1e-16)
                
                # Cycles are plotted with colors transitioning across the set
                ax.plot(t_normalized, cycle_signals[i], 'o-', markersize=2, 
                       label='Cycle {}'.format(i), color=colors[i], alpha=0.7)
            
            ax.set_xlabel('Normalized time within cycle')
            ax.set_ylabel('Signal value')
            ax.set_title('Cycle Overlay: {}'.format(signal_name))
            ax.legend()
            # A grid is added for readability
            ax.grid(True, alpha=0.3)
            
            # File name is sanitized for safe storage
            safe_name = signal_name.replace('/', '_')
            plt.savefig(output_dir / "signal_overlay_{}.png".format(safe_name), 
                       dpi=100, bbox_inches='tight')
            plt.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Numerical validation of FirstBlood simulation results'
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (e.g., patient_025)')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Full path to results directory')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    results_base = project_root / 'projects' / 'simple_run' / 'results'
    
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.model:
        results_dir = results_base / args.model
    else:
        model_name = select_model_interactive(results_base)
        if model_name is None:
            print("No model selected.")
            sys.exit(1)
        results_dir = results_base / model_name
    
    if not results_dir.exists():
        print("Error: Results directory not found: {}".format(results_dir))
        sys.exit(1)
    
    model_name = results_dir.name
    output_dir = project_root / 'pipeline' / 'output' / model_name / 'numerical_validation'
    

    print("Model: {}".format(model_name))
    print("Results: {}".format(results_dir))
    print("Output: {}".format(output_dir))
    print("")
    
    files = discover_timeseries_files(results_dir)
    print("Found {} timeseries files".format(len(files)))
    

    integrity_checks = []
    for f in files[:100]:
        data = load_timeseries(f)
        if data is not None:
            check = check_integrity(data, f)
            integrity_checks.append(check)
    
    print("Checked {} files".format(len(integrity_checks)))
    
    convergence_results, all_cycles_data = analyze_convergence(results_dir)
    print("Found convergence data for {} signals".format(len(convergence_results)))
    
    write_reports(output_dir, integrity_checks, convergence_results, model_name)
    
    plot_diagnostics(output_dir, all_cycles_data)
    

    print("Output directory: {}".format(output_dir))
    print("  - numerical_stability_report.txt")
    print("  - numerical_stability_report.json")
    print("  - signal_overlay_*.png (if data available)")
    print("")


if __name__ == "__main__":
    main()