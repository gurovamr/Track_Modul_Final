#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


PRESSURE_BASELINE = 101325
FLOW_UNIT_THRESHOLD = 0.1

GLOBAL_REFERENCE_RANGES = {
    'SBP': '90-120 mmHg',
    'DBP': '60-80 mmHg',
    'MAP': '70-100 mmHg',
    'HR': '60-100 bpm',
    'SV': '60-100 mL',
    'CO': '4.0-8.0 L/min',
}


@dataclass
class HemodynamicMetrics:
    HR: Optional[float]
    cycle_period: Optional[float]
    P_sys: Optional[float]
    P_dia: Optional[float]
    P_map: Optional[float]
    SV_ml: Optional[float]
    CO_lmin: Optional[float]
    rms_convergence_pct: Optional[float]
    t_total: float
    n_steps: int
    dt_mean: float
    pressure_file_used: Optional[str] = None
    pressure_col_idx: Optional[int] = None
    co_file_used: Optional[str] = None
    co_col_idx: Optional[int] = None


def list_models(results_base_path: Path) -> List[str]:
    """
    Finds all model directories in the results folder and returns them sorted.
    """
    if not results_base_path.exists():
        return []
    
    models = [d.name for d in results_base_path.iterdir() 
              if d.is_dir()]
    return sorted(models)


def select_model_interactive(results_base_path: Path) -> Optional[str]:
    """
    Shows available models and lets the user pick one. Returns the chosen model name or None.
    """
    models = list_models(results_base_path)
    
    if not models:
        print("No model directories found.")
        return None
    
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print("  {}. {}".format(i, model))
    
    while True:
        selection = input("\nSelect model to analyze (type name or number): ").strip()
        
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print("Invalid number. ")
        elif selection in models:
            return selection
        else:
            print("Model not found.")


def discover_txt_files(results_dir: Path) -> List[Path]:
    """
    Finds all .txt files recursively and returns them sorted.
    """
    txt_files = []
    
    for item in results_dir.rglob('*.txt'):
        txt_files.append(item)
    
    return sorted(txt_files)


def load_timeseries(filepath: Path) -> Optional[np.ndarray]:
    """
    Loads timeseries data from a text file. Tries comma-separated values first, then falls back to whitespace.
    """
    try:
        data = np.genfromtxt(filepath, delimiter=',', max_rows=None)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[0] > 1:
            return data
    except Exception:
        pass
    
    try:
        data = np.genfromtxt(filepath, delimiter=None, max_rows=None)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[0] > 1:
            return data
    except Exception:
        pass
    
    return None


def detect_time_column(data: np.ndarray) -> Optional[int]:
    """
    Finds which column has strictly increasing time values. Returns the column index or None if not found.
    """
    if data.ndim != 2 or data.shape[0] < 10:
        return None
    
    for col_idx in range(data.shape[1]):
        col = data[:, col_idx]
        
        if np.any(np.isnan(col)) or np.any(np.isinf(col)):
            continue
        
        diffs = np.diff(col)
        
        if np.all(diffs > 1e-12) and np.mean(diffs) > 0:
            return col_idx
    
    return None


def select_pulsatile_column(data: np.ndarray, time_col_idx: int) -> Optional[int]:
    """
    Picks the column with the biggest oscillations (largest peak-to-peak). Skips the time column and flat signals.
    """
    if data.ndim != 2 or data.shape[0] < 10:
        return None
    
    best_col_idx = None
    best_amplitude = 0.0
    
    for col_idx in range(data.shape[1]):
        if col_idx == time_col_idx:
            continue
        
        col = data[:, col_idx]
        
        if np.any(np.isnan(col)) or np.any(np.isinf(col)):
            continue
        
        col_std = np.std(col)
        if col_std < 1e-10:
            continue
        
        amplitude = np.max(col) - np.min(col)
        
        if amplitude > best_amplitude:
            best_amplitude = amplitude
            best_col_idx = col_idx
    
    return best_col_idx


def select_volumetric_flow_column(data: np.ndarray, time_col_idx: int) -> Optional[int]:
    """
    Selects a volumetric flow column from common output formats.
    """
    if data.ndim != 2 or data.shape[0] < 10:
        return None
    ncols = data.shape[1]
    if ncols >= 13:
        candidates = [5, 6]
        best_idx = None
        best_std = -1.0
        for idx in candidates:
            col = data[:, idx]
            if np.any(np.isnan(col)) or np.any(np.isinf(col)):
                continue
            col_std = float(np.nanstd(col))
            if col_std > best_std:
                best_std = col_std
                best_idx = idx
        if best_idx is not None:
            return best_idx
        return 5
    if ncols == 3:
        return 2
    if ncols == 2:
        return 1
    candidates = [i for i in range(ncols) if i != time_col_idx]
    if not candidates:
        return None
    pressure_like = []
    for idx in candidates:
        col = data[:, idx]
        if np.any(np.isnan(col)) or np.any(np.isinf(col)):
            continue
        if np.nanmedian(np.abs(col)) > 1e4:
            pressure_like.append(idx)
    fallback = [i for i in candidates if i not in pressure_like]
    if fallback:
        return fallback[0]
    return candidates[0]


def extract_volumetric_flow_series(data: np.ndarray, time_col_idx: int) -> Tuple[Optional[np.ndarray], List[int], str]:
    """
    Extracts a volumetric flow series from common FirstBlood output formats.
    """
    if data.ndim != 2 or data.shape[0] < 10:
        return None, [], "invalid data"
    ncols = data.shape[1]
    if ncols >= 13:
        vfrs = data[:, 5]
        vfre = data[:, 6]
        flow = 0.5 * (vfrs + vfre)
        return flow, [5, 6], "avg(volume_flow_rate_start, volume_flow_rate_end)"
    if ncols == 3:
        return data[:, 2], [2], "node volume_flow_rate"
    if ncols == 2:
        return data[:, 1], [1], "edge volume_flow_rate"
    flow_idx = select_volumetric_flow_column(data, time_col_idx)
    if flow_idx is None:
        return None, [], "no flow column detected"
    return data[:, flow_idx], [flow_idx], "heuristic flow column"


def detect_flow_unit(flow_cycle: np.ndarray) -> Tuple[str, float]:
    """
    Detects flow units based on magnitude and returns unit label and scale to mL/min.
    """
    max_abs = float(np.nanmax(np.abs(flow_cycle)))
    if max_abs < 1e-3:
        return "m^3/s", 1e6 * 60.0
    if max_abs < 1e3:
        return "mL/s", 60.0
    return "mL/min", 1.0


def print_flow_file_debug(vessel_name: str, data: np.ndarray) -> None:
    """
    Prints detailed debug info for a cerebral vessel file.
    """
    print("\n[Debug] {} file diagnostics".format(vessel_name))
    print("  Array shape: {}".format(data.shape))
    print("  First 5 rows:\n{}".format(np.array2string(data[:5], precision=6)))
    for col_idx in range(data.shape[1]):
        col = data[:, col_idx]
        col_min = float(np.nanmin(col))
        col_max = float(np.nanmax(col))
        print("  Col {:>2d} min/max: {:.6e} / {:.6e}".format(col_idx, col_min, col_max))


def choose_aortic_signals(results_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Looks for aortic pressure files. Checks the standard location first, then searches by filename patterns.
    """
    primary_aorta = results_dir / 'heart_kim_lit' / 'aorta.txt'
    
    if primary_aorta.exists():
        return primary_aorta, None, None
    
    files = discover_txt_files(results_dir)
    
    aorta_candidates = []
    for f in files:
        path_lower = str(f).lower()
        if any(x in path_lower for x in ['aorta', 'ao_', 'aortic', 'root']):
            aorta_candidates.append(f)
    
    if not aorta_candidates:
        print("Warning: No aortic signals found in results.")
        print("Searched for: heart_kim_lit/aorta.txt and files containing: aorta, ao_, aortic, root")
        return None, None, None
    
    return aorta_candidates[0], None, None


def resample_uniform(time_col: np.ndarray, signal_col: np.ndarray, 
                     n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples the signal onto a uniform time grid via linear interpolation.
    """
    t_uniform = np.linspace(time_col[0], time_col[-1], n_points)
    s_uniform = np.interp(t_uniform, time_col, signal_col)
    
    return t_uniform, s_uniform


def numpy_find_peaks(signal: np.ndarray, distance: int = 5, height: float = 0.0) -> np.ndarray:
    """
    Finds peaks (local maxima) in the signal using only numpy. Returns the indices of the peaks.
    """
    peaks = []
    n = len(signal)
    
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if signal[i] >= height:
                if not peaks or (i - peaks[-1]) >= distance:
                    peaks.append(i)
    
    return np.array(peaks)


def estimate_period_autocorr(signal_data: np.ndarray, 
                             time_data: np.ndarray,
                             min_period: float = 0.4,
                             max_period: float = 1.5) -> Optional[float]:
    """
    Estimates the cardiac period by looking for repeating patterns in the signal using autocorrelation.
    """
    if len(signal_data) < 50:
        return None
    
    late_idx = max(0, int(0.7 * len(signal_data)))
    late_signal = signal_data[late_idx:]
    late_time = time_data[late_idx:]
    
    late_signal = late_signal - np.mean(late_signal)
    late_signal = late_signal / (np.std(late_signal) + 1e-16)
    
    acf = np.correlate(late_signal, late_signal, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / (acf[0] + 1e-16)
    
    min_lag_samples = max(5, int(min_period / (np.mean(np.diff(late_time)) + 1e-16)))
    max_lag_samples = int(max_period / (np.mean(np.diff(late_time)) + 1e-16))
    
    if min_lag_samples >= len(acf):
        return None
    
    max_lag_samples = min(max_lag_samples, len(acf) - 1)
    
    acf_window = acf[min_lag_samples:max_lag_samples]
    
    peaks = numpy_find_peaks(acf_window, distance=5, height=0.2)
    
    if len(peaks) == 0:
        return None
    
    peak_heights = acf_window[peaks]
    best_peak_idx = peaks[np.argmax(peak_heights)]
    lag_samples = best_peak_idx + min_lag_samples
    
    mean_dt = np.mean(np.diff(late_time))
    period = lag_samples * mean_dt
    
    return period


def extract_last_cycle(signal_data: np.ndarray, 
                       time_data: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Pulls out the last complete cardiac cycle from the signal by finding peaks.
    """
    if signal_data is None or len(signal_data) < 20:
        return None, None
    
    prominence_threshold = np.max(signal_data) * 0.05
    peaks = numpy_find_peaks(signal_data, distance=5, height=prominence_threshold)
    
    if len(peaks) < 2:
        return None, None
    
    start_idx = peaks[-2]
    end_idx = peaks[-1]
    
    end_idx = min(end_idx + 1, len(signal_data))
    
    cycle_signal = signal_data[start_idx:end_idx]
    cycle_time = time_data[start_idx:end_idx]
    
    return cycle_signal, cycle_time


def extract_two_cycles(signal_data: np.ndarray, 
                       time_data: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                                        Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extracts the last two complete cardiac cycles from the signal.
    """
    if signal_data is None or len(signal_data) < 20:
        return None, None, None, None
    
    prominence_threshold = np.max(signal_data) * 0.05
    peaks = numpy_find_peaks(signal_data, distance=5, height=prominence_threshold)
    
    if len(peaks) < 3:
        return None, None, None, None
    
    start1_idx = peaks[-3]
    end1_idx = peaks[-2]
    end2_idx = peaks[-1]
    
    end1_idx = min(end1_idx + 1, len(signal_data))
    end2_idx = min(end2_idx + 1, len(signal_data))
    
    cycle1_signal = signal_data[start1_idx:end1_idx]
    cycle1_time = time_data[start1_idx:end1_idx]
    cycle2_signal = signal_data[end1_idx:end2_idx]
    cycle2_time = time_data[end1_idx:end2_idx]
    
    return cycle1_signal, cycle1_time, cycle2_signal, cycle2_time


def compute_pressure_metrics(signal_data: np.ndarray) -> Dict[str, float]:
    """
    Calculates systolic, diastolic, and mean pressure from a cycle (assuming the signal is already in mmHg).
    """
    if signal_data is None or len(signal_data) < 2:
        return {}
    
    return {
        'P_sys': float(np.max(signal_data)),
        'P_dia': float(np.min(signal_data)),
        'P_map': float(np.mean(signal_data)),
    }


def compute_flow_metrics(velocities: np.ndarray, 
                        diameters: np.ndarray,
                        time_col: np.ndarray) -> Dict[str, float]:
    """
    Calculates stroke volume and cardiac output from velocity and diameter using Q = v * A.
    """
    if velocities is None or diameters is None or time_col is None:
        return {}
    
    radii = diameters / 2.0
    areas = np.pi * radii * radii
    flows = velocities * areas
    
    sv_m3 = np.trapz(flows, time_col)
    sv_ml = sv_m3 * 1e6
    
    cycle_period = time_col[-1] - time_col[0]
    co_lmin = (sv_ml / cycle_period) * (60.0 / 1000.0)
    
    return {
        'SV_ml': float(sv_ml),
        'CO_lmin': float(co_lmin),
    }


def compute_convergence_rms(cycle1: np.ndarray, cycle2: np.ndarray) -> float:
    """
    Calculates how different two cycles are as a percentage using RMS.
    """
    if cycle1 is None or cycle2 is None or len(cycle1) < 2 or len(cycle2) < 2:
        return None
    
    n = max(len(cycle1), len(cycle2))
    t1 = np.linspace(0, 1, len(cycle1))
    t2 = np.linspace(0, 1, len(cycle2))
    t_common = np.linspace(0, 1, n)
    
    c1_interp = np.interp(t_common, t1, cycle1)
    c2_interp = np.interp(t_common, t2, cycle2)
    
    mse = np.mean((c1_interp - c2_interp)**2)
    rms = np.sqrt(mse)
    
    p_range = np.max(cycle2) - np.min(cycle2)
    if p_range < 1e-12:
        return None
    
    rms_pct = (rms / p_range) * 100.0
    
    return rms_pct


def compute_cerebral_flow_summary(results_dir: Path,
                                  output_dir: Path,
                                  cycle_period: Optional[float],
                                  co_lmin: Optional[float]) -> Tuple[Dict[str, float], Optional[float], Optional[float]]:
    """
    Calculates mean flow rates for key cerebral vessels over the last cardiac cycle and saves to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vessel_map = {
        'L_ICA': 'A16',
        'R_ICA': 'A12',
        'Basilar': 'A56',
        'ACoA': 'A77',
        'L_PCoA': 'A63',
        'R_PCoA': 'A62',
    }
    
    cerebral_flows = {}
    
    for vessel_name, vessel_id in vessel_map.items():
        flow_file = results_dir / 'arterial' / f'{vessel_id}.txt'
        
        if not flow_file.exists():
            cerebral_flows[vessel_name] = np.nan
            continue
        
        try:
            data = load_timeseries(flow_file)
            if data is None or data.shape[0] < 10:
                cerebral_flows[vessel_name] = np.nan
                continue

            print_flow_file_debug(vessel_name, data)
            
            time_idx = detect_time_column(data)
            if time_idx is None:
                time_idx = 0
            
            flow_col, flow_cols, flow_note = extract_volumetric_flow_series(data, time_idx)
            if flow_col is None:
                cerebral_flows[vessel_name] = np.nan
                continue
            
            time_col = data[:, time_idx]
            flow_cycle, _flow_time_cycle = extract_last_cycle(flow_col, time_col)
            if flow_cycle is None or len(flow_cycle) < 2:
                if cycle_period is not None and cycle_period > 0:
                    t_end = time_col[-1]
                    t_start = t_end - cycle_period
                    mask = (time_col >= t_start) & (time_col <= t_end)
                    flow_cycle = flow_col[mask]
                if flow_cycle is None or len(flow_cycle) < 2:
                    cerebral_flows[vessel_name] = np.nan
                    continue
            
            mean_flow = float(np.mean(flow_cycle))
            unit_label, unit_scale = detect_flow_unit(flow_cycle)
            mean_flow_mlmin = mean_flow * unit_scale
            cerebral_flows[vessel_name] = mean_flow_mlmin
            
            print("\n[Debug] {} flow selection".format(vessel_name))
            print("  Flow column(s): {}".format(flow_cols))
            print("  Flow selection: {}".format(flow_note))
            print("  Unit assumption: {}".format(unit_label))
            print("  First 5 samples: {}".format(np.array2string(flow_cycle[:5], precision=6)))
            print("  Min: {:.6e}  Max: {:.6e}".format(float(np.min(flow_cycle)), float(np.max(flow_cycle))))
            
        except Exception:
            cerebral_flows[vessel_name] = np.nan
    
    q_brain_mlmin = None
    brain_fraction = None
    q_terms = [
        cerebral_flows.get('L_ICA', np.nan),
        cerebral_flows.get('R_ICA', np.nan),
        cerebral_flows.get('Basilar', np.nan),
    ]
    if any(isinstance(x, float) and np.isnan(x) for x in q_terms):
        q_brain_mlmin = np.nan
    else:
        q_brain_mlmin = float(np.sum(q_terms))
    if co_lmin is not None and not (isinstance(co_lmin, float) and np.isnan(co_lmin)):
        if q_brain_mlmin is not None and not (isinstance(q_brain_mlmin, float) and np.isnan(q_brain_mlmin)):
            brain_fraction = q_brain_mlmin / (co_lmin * 1000.0)
            if brain_fraction < 0.05 or brain_fraction > 0.30:
                print("\nWarning: brain_fraction {:.4f} outside expected range [0.05, 0.30]".format(brain_fraction))

    df = pd.DataFrame([{
        **cerebral_flows,
        'Q_brain_mlmin': q_brain_mlmin,
        'brain_fraction': brain_fraction,
    }])
    csv_file = output_dir / 'cerebral_flow_summary.csv'
    df.to_csv(csv_file, index=False)
    
    plot_cerebral_flow_barchart(cerebral_flows, output_dir)
    
    return cerebral_flows, q_brain_mlmin, brain_fraction


def plot_cerebral_flow_barchart(cerebral_flows: Dict[str, float], output_dir: Path) -> None:
    """
    Saves a bar chart of mean cerebral vessel flows (mL/min).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ['L_ICA', 'R_ICA', 'Basilar', 'ACoA', 'L_PCoA', 'R_PCoA']
    values = []
    colors = []
    for label in labels:
        value = cerebral_flows.get(label, np.nan)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            values.append(0.0)
            colors.append('lightgray')
        else:
            values.append(value)
            colors.append('tab:blue')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Mean Flow (mL/min)', fontsize=12)
    ax.set_title('Cerebral Flow Distribution - Last Cycle', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, label in zip(bars, labels):
        value = cerebral_flows.get(label, np.nan)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            ax.text(bar.get_x() + bar.get_width() / 2.0, 0.0, 'N/A',
                    ha='center', va='bottom', fontsize=9, color='gray')
        else:
            y_pos = bar.get_height()
            va = 'bottom' if y_pos >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, '{:.1f}'.format(value),
                    ha='center', va=va, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cerebral_flow_barchart.png', dpi=100, bbox_inches='tight')
    plt.close()


def convert_flow_series_to_ml_s(flow_values: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Converts a flow series to mL/s using a simple unit heuristic.
    """
    if flow_values is None or len(flow_values) == 0:
        return flow_values, 'N/A'
    max_abs = np.nanmax(np.abs(flow_values))
    if max_abs > FLOW_UNIT_THRESHOLD:
        return flow_values, 'assumed mL/s'
    return flow_values * 1e6, 'assumed m^3/s'


def format_value(value: Optional[float], fmt: str, suffix: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return (fmt.format(value) + suffix).strip()


def build_global_summary_lines(metrics: HemodynamicMetrics) -> List[str]:
    lines = []
    lines.append("GLOBAL HEMODYNAMIC SUMMARY")
    lines.append("-" * 80)
    lines.append("SBP: {:>12s}  (ref {})".format(
        format_value(metrics.P_sys, "{:.1f}", " mmHg"), GLOBAL_REFERENCE_RANGES['SBP']))
    lines.append("DBP: {:>12s}  (ref {})".format(
        format_value(metrics.P_dia, "{:.1f}", " mmHg"), GLOBAL_REFERENCE_RANGES['DBP']))
    lines.append("MAP: {:>12s}  (ref {})".format(
        format_value(metrics.P_map, "{:.1f}", " mmHg"), GLOBAL_REFERENCE_RANGES['MAP']))
    lines.append("HR : {:>12s}  (ref {})".format(
        format_value(metrics.HR, "{:.1f}", " bpm"), GLOBAL_REFERENCE_RANGES['HR']))
    lines.append("SV : {:>12s}  (ref {})".format(
        format_value(metrics.SV_ml, "{:.1f}", " mL"), GLOBAL_REFERENCE_RANGES['SV']))
    lines.append("CO : {:>12s}  (ref {})".format(
        format_value(metrics.CO_lmin, "{:.2f}", " L/min"), GLOBAL_REFERENCE_RANGES['CO']))
    lines.append("")
    return lines


def build_cerebral_summary_lines(cerebral_flows: Dict[str, float],
                                 q_brain_mlmin: Optional[float],
                                 brain_fraction: Optional[float]) -> List[str]:
    lines = []
    lines.append("CEREBRAL HEMODYNAMIC DISTRIBUTION")
    lines.append("-" * 80)
    for vessel_name, flow in cerebral_flows.items():
        if flow is None or (isinstance(flow, float) and np.isnan(flow)):
            lines.append("{:<12s}: N/A".format(vessel_name))
        else:
            lines.append("{:<12s}: {:>8.2f} mL/min".format(vessel_name, flow))
    lines.append("Q_brain:     {}".format(format_value(q_brain_mlmin, "{:.2f}", " mL/min")))
    if brain_fraction is None or (isinstance(brain_fraction, float) and np.isnan(brain_fraction)):
        lines.append("Brain/CO:    N/A")
    else:
        lines.append("Brain/CO:    {:.4f}".format(brain_fraction))
    lines.append("")
    return lines


def write_summary_and_csv(metrics: HemodynamicMetrics,
                          output_dir: Path,
                          cerebral_flows: Dict[str, float],
                          q_brain_mlmin: Optional[float],
                          brain_fraction: Optional[float]):
    """
    Generates a readable summary report and exports the metrics to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("=" * 80)
    lines.append("FIRSTBLOOD BIOLOGICAL VALIDATION ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("DATA SOURCES")
    lines.append("-" * 80)
    if metrics.pressure_file_used:
        lines.append("Aortic pressure file: {}".format(metrics.pressure_file_used))
        if metrics.pressure_col_idx is not None:
            lines.append("  Column index: {}".format(metrics.pressure_col_idx))
    if metrics.co_file_used:
        lines.append("Cardiac output file:  {}".format(metrics.co_file_used))
        if metrics.co_col_idx is not None:
            lines.append("  Column index: {}".format(metrics.co_col_idx))
    else:
        lines.append("Cardiac output file:  Not available (SV and CO not computed)")
    lines.append("")
    
    lines.append("SIMULATION METADATA")
    lines.append("-" * 80)
    lines.append("Total simulation time: {:.3f} s".format(metrics.t_total))
    lines.append("Number of time steps:  {}".format(metrics.n_steps))
    lines.append("Mean time step (dt):   {:.6e} s".format(metrics.dt_mean))
    lines.append("")
    
    lines.extend(build_global_summary_lines(metrics))
    if metrics.cycle_period is not None:
        lines.append("Cycle period:          {:.4f} s".format(metrics.cycle_period))
    if metrics.rms_convergence_pct is not None:
        lines.append("Convergence RMS%:      {:.4f} %".format(metrics.rms_convergence_pct))
    lines.append("")
    lines.extend(build_cerebral_summary_lines(cerebral_flows, q_brain_mlmin, brain_fraction))
    lines.append("=" * 80)
    lines.append("OUTPUT FILES GENERATED")
    lines.append("=" * 80)
    lines.append("  - global_metrics.csv")
    lines.append("  - cerebral_flow_summary.csv")
    lines.append("  - aortic_pressure_last_cycle.png")
    lines.append("  - aortic_flow_last_cycle.png (if data available)")
    lines.append("  - cycle_overlay_aortic_pressure.png")
    lines.append("  - cerebral_flow_barchart.png")
    lines.append("=" * 80)
    lines.append("")
    
    text_report = "\n".join(lines)
    
    report_file = output_dir / "biological_validation_summary.txt"
    with open(report_file, 'w') as f:
        f.write(text_report)
    
    metrics_dict = {
        'HR_bpm': metrics.HR,
        'cycle_period_s': metrics.cycle_period,
        'P_sys_mmHg': metrics.P_sys,
        'P_dia_mmHg': metrics.P_dia,
        'P_map_mmHg': metrics.P_map,
        'SV_ml': metrics.SV_ml,
        'CO_lmin': metrics.CO_lmin,
        'rms_convergence_pct': metrics.rms_convergence_pct,
    }
    
    metrics_df = pd.DataFrame([metrics_dict])
    csv_file = output_dir / "global_metrics.csv"
    metrics_df.to_csv(csv_file, index=False)


def make_plots(results_dir: Path, output_dir: Path,
               pressure_file: Path,
               aortic_flow_file: Optional[Path] = None):
    """
    Creates visualization plots showing pressure cycles and convergence.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_p = load_timeseries(pressure_file)
    if data_p is None:
        return
    
    time_col_idx = detect_time_column(data_p)
    if time_col_idx is None:
        return
    
    pressure_col_idx = select_pulsatile_column(data_p, time_col_idx)
    if pressure_col_idx is None:
        return
    
    time_col = data_p[:, time_col_idx]
    pressure_col = data_p[:, pressure_col_idx]
    
    pressure_col = (pressure_col - PRESSURE_BASELINE) / 133.322
    
    try:
        p_cycle, t_cycle = extract_last_cycle(pressure_col, time_col)
        
        if p_cycle is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(t_cycle, p_cycle, 'b-', linewidth=2)
            ax.fill_between(t_cycle, p_cycle, alpha=0.3)
            sbp = np.max(p_cycle)
            dbp = np.min(p_cycle)
            ax.axhline(sbp, color='red', linestyle='--', linewidth=1, label='SBP')
            ax.axhline(dbp, color='green', linestyle='--', linewidth=1, label='DBP')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Pressure (mmHg)', fontsize=12)
            ax.set_title('Aortic Pressure - Last Cycle', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.savefig(output_dir / "aortic_pressure_last_cycle.png", 
                       dpi=100, bbox_inches='tight')
            plt.close()
    except Exception:
        pass

    if aortic_flow_file is not None and aortic_flow_file.exists():
        data_flow = load_timeseries(aortic_flow_file)
        if data_flow is not None:
            flow_time_idx = detect_time_column(data_flow)
            if flow_time_idx is not None:
                flow_col_idx = select_pulsatile_column(data_flow, flow_time_idx)
                if flow_col_idx is not None:
                    flow_time = data_flow[:, flow_time_idx]
                    flow_col = data_flow[:, flow_col_idx]
                    flow_cycle, flow_t_cycle = extract_last_cycle(flow_col, flow_time)
                    if flow_cycle is not None and flow_t_cycle is not None:
                        flow_cycle_ml_s, unit_note = convert_flow_series_to_ml_s(flow_cycle)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(flow_t_cycle, flow_cycle_ml_s, color='tab:blue', linewidth=2)
                        ax.fill_between(flow_t_cycle, 0, flow_cycle_ml_s,
                                        where=flow_cycle_ml_s >= 0,
                                        color='tab:blue', alpha=0.3, label='Forward SV')
                        ax.fill_between(flow_t_cycle, 0, flow_cycle_ml_s,
                                        where=flow_cycle_ml_s < 0,
                                        color='tab:red', alpha=0.3, label='Reverse SV')
                        ax.set_xlabel('Time (s)', fontsize=12)
                        ax.set_ylabel('Flow (mL/s)', fontsize=12)
                        ax.set_title('Aortic Flow - Last Cycle', fontsize=13)
                        ax.legend(fontsize=10)
                        ax.grid(True, alpha=0.3)
                        plt.savefig(output_dir / "aortic_flow_last_cycle.png",
                                   dpi=100, bbox_inches='tight')
                        plt.close()
    
    try:
        c1, t1, c2, t2 = extract_two_cycles(pressure_col, time_col)
        
        if c1 is not None and c2 is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            t1_norm = (t1 - t1[0]) / (t1[-1] - t1[0] + 1e-16)
            t2_norm = (t2 - t2[0]) / (t2[-1] - t2[0] + 1e-16)
            
            ax.plot(t1_norm, c1, 'o-', markersize=1, label='Cycle n-1', 
                   color='blue', alpha=0.7, linewidth=1)
            ax.plot(t2_norm, c2, 'o-', markersize=1, label='Cycle n', 
                   color='red', alpha=0.7, linewidth=1)
            
            ax.set_xlabel('Normalized Cycle Time', fontsize=12)
            ax.set_ylabel('Pressure (mmHg)', fontsize=12)
            ax.set_title('Cycle Overlay - Last Two Cycles', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.savefig(output_dir / "cycle_overlay_aortic_pressure.png", 
                       dpi=100, bbox_inches='tight')
            plt.close()
    except Exception:
        pass


def main():
    """
    Main entry point that handles both interactive mode and command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Biological validation of FirstBlood simulation'
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Model name')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Full path to results directory')
    parser.add_argument('--output_root', type=str, default=None,
                       help='Output root directory')
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
        print(": Results directory not found: {}".format(results_dir))
        sys.exit(1)
    
    model_name = results_dir.name
    
    if args.output_root:
        output_root = Path(args.output_root)
        output_dir = output_root / model_name / 'biological_analysis'
    else:
        output_dir = project_root / 'pipeline' / 'output' / model_name / 'biological_analysis'
    
    print("")
    print("=" * 80)
    print("BIOLOGICAL VALIDATION ANALYSIS")
    print("=" * 80)
    print("Model:        {}".format(model_name))
    print("Results dir:  {}".format(results_dir))
    print("Output dir:   {}".format(output_dir))
    print("")
    
    pressure_file, _velocity_file, _diameter_file = choose_aortic_signals(results_dir)
    
    if pressure_file is None:
        print("Error: Could not locate aortic signals.")
        sys.exit(1)
    
    print("Found aortic pressure: {}".format(pressure_file.name))
    
    data_p = load_timeseries(pressure_file)
    if data_p is None:
        print("Error: Could not load pressure data.")
        sys.exit(1)
    
    time_col_idx = detect_time_column(data_p)
    if time_col_idx is None:
        print("Error: Could not detect time column.")
        sys.exit(1)
    
    pressure_col_idx = select_pulsatile_column(data_p, time_col_idx)
    if pressure_col_idx is None:
        print("Error: Could not detect pulsatile pressure column.")
        sys.exit(1)
    
    print("Selected pressure column index: {}".format(pressure_col_idx))
    
    time_col = data_p[:, time_col_idx]
    pressure_col = data_p[:, pressure_col_idx]
    
    pressure_col = (pressure_col - PRESSURE_BASELINE) / 133.322
    
    dt_stats = np.diff(time_col)
    dt_mean = float(np.mean(dt_stats))
    t_total = float(time_col[-1])
    n_steps = len(time_col)
    
    period = estimate_period_autocorr(pressure_col, time_col)
    
    p_cycle, t_cycle = extract_last_cycle(pressure_col, time_col)
    
    sv_ml = None
    co_lmin = None
    co_file_used = None
    co_col_idx = None
    
    co_file = results_dir / 'heart_kim_lit' / 'L_lv_aorta.txt'
    if co_file.exists():
        print("\nComputing cardiac output from: {}".format(co_file.name))
        data_co = load_timeseries(co_file)
        if data_co is not None:
            co_time_col_idx = detect_time_column(data_co)
            if co_time_col_idx is not None:
                co_signal_idx = select_pulsatile_column(data_co, co_time_col_idx)
                if co_signal_idx is not None:
                    print("Selected CO flow column index: {}".format(co_signal_idx))
                    co_time = data_co[:, co_time_col_idx]
                    co_flow = data_co[:, co_signal_idx]
                    
                    co_cycle, co_t_cycle = extract_last_cycle(co_flow, co_time)
                    if co_cycle is not None and co_t_cycle is not None:
                        co_cycle_ml_s, unit_note = convert_flow_series_to_ml_s(co_cycle)
                        sv_ml = float(np.trapz(co_cycle_ml_s, co_t_cycle))
                        
                        if period is not None and period > 0:
                            co_lmin = (sv_ml / 1000.0) * (60.0 / period)
                        
                        co_file_used = str(co_file.relative_to(results_dir))
                        co_col_idx = co_signal_idx
                        print("  SV: {:.1f} mL".format(sv_ml))
                        print("  CO: {:.2f} L/min".format(co_lmin) if co_lmin else "  CO: N/A")
    
    if co_file_used is None:
        print("\nCardiac output file not available: {}".format(co_file))
    
    metrics = HemodynamicMetrics(
        HR=60.0 / period if period is not None else None,
        cycle_period=period,
        P_sys=float(np.max(p_cycle)) if p_cycle is not None else None,
        P_dia=float(np.min(p_cycle)) if p_cycle is not None else None,
        P_map=float(np.mean(p_cycle)) if p_cycle is not None else None,
        SV_ml=sv_ml,
        CO_lmin=co_lmin,
        rms_convergence_pct=None,
        t_total=t_total,
        n_steps=n_steps,
        dt_mean=dt_mean,
        pressure_file_used=str(pressure_file.relative_to(results_dir)),
        pressure_col_idx=pressure_col_idx,
        co_file_used=co_file_used,
        co_col_idx=co_col_idx,
    )
    
    c1, t1, c2, t2 = extract_two_cycles(pressure_col, time_col)
    if c1 is not None and c2 is not None:
        rms_pct = compute_convergence_rms(c1, c2)
        metrics.rms_convergence_pct = rms_pct
    
    print("\nMetrics computed:")
    print("  HR: {:.1f} bpm".format(metrics.HR) if metrics.HR else "  HR: N/A")
    print("  Psys: {:.1f} mmHg".format(metrics.P_sys) if metrics.P_sys else "  Psys: N/A")
    print("  Pdia: {:.1f} mmHg".format(metrics.P_dia) if metrics.P_dia else "  Pdia: N/A")
    print("  Pmap: {:.1f} mmHg".format(metrics.P_map) if metrics.P_map else "  Pmap: N/A")
    print("  SV: {:.1f} mL".format(metrics.SV_ml) if metrics.SV_ml else "  SV: N/A")
    print("  CO: {:.2f} L/min".format(metrics.CO_lmin) if metrics.CO_lmin else "  CO: N/A")
    if metrics.rms_convergence_pct is not None:
        print("  Convergence RMS%: {:.4f}".format(metrics.rms_convergence_pct))

    cerebral_flows, q_brain_mlmin, brain_fraction = compute_cerebral_flow_summary(
        results_dir,
        output_dir,
        metrics.cycle_period,
        metrics.CO_lmin,
    )
    
    print("\nGlobal hemodynamic summary:")
    for line in build_global_summary_lines(metrics)[2:-1]:
        print("  {}".format(line))
    print("\nCerebral hemodynamic distribution (mL/min):")
    for line in build_cerebral_summary_lines(cerebral_flows, q_brain_mlmin, brain_fraction)[2:-1]:
        print("  {}".format(line))
    
    print("\nWriting outputs...")
    write_summary_and_csv(metrics, output_dir, cerebral_flows, q_brain_mlmin, brain_fraction)
    make_plots(results_dir, output_dir, pressure_file, co_file)
    
    print("")
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Output directory: {}".format(output_dir))
    print("")


if __name__ == "__main__":
    main()
