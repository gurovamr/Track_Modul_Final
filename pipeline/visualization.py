#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import csv
from typing import Optional, List, Tuple, Dict
from collections import deque, defaultdict
import warnings

warnings.filterwarnings('ignore')


PRESSURE_BASELINE = 101325


def list_models(results_base_path: Path) -> List[str]:
    """
    Finds all available model directories and returns them sorted by name.
    """
    if not results_base_path.exists():
        return []
    
    models = [d.name for d in results_base_path.iterdir() 
              if d.is_dir()]
    return sorted(models)


def select_model_interactive(results_base_path: Path) -> Optional[str]:
    """
    Shows the user a numbered list of available models and lets them pick one by name or number.
    """
    models = list_models(results_base_path)
    
    if not models:
        print("No model directories found.")
        return None
    
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print("  {}. {}".format(i, model))
    
    while True:
        selection = input("\nSelect model to visualize (type name or number): ").strip()
        
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print("Invalid number. Try again.")
        elif selection in models:
            return selection
        else:
            print("Model not found. Try again.")


def load_timeseries(filepath: Path) -> Optional[np.ndarray]:
    """
    Attempts to load timeseries data from a text file. First tries comma-separated values,
    then falls back to whitespace-delimited parsing. Returns None if both fail.
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
    Identifies which column contains monotonically increasing time values.
    Returns the column index, or None if no suitable time column is found.
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
    Finds the column with the largest peak-to-peak variation (excluding the time column).
    This is typically the physiological signal of interest. Returns None if no good candidate exists.
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


def resample_uniform(time_col: np.ndarray, signal_col: np.ndarray, 
                     n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates the signal onto a uniform time grid using linear interpolation.
    This creates evenly-spaced samples for cleaner plotting and analysis.
    """
    t_uniform = np.linspace(time_col[0], time_col[-1], n_points)
    s_uniform = np.interp(t_uniform, time_col, signal_col)
    
    return t_uniform, s_uniform


def numpy_find_peaks(signal: np.ndarray, distance: int = 5, height: float = 0.0) -> np.ndarray:
    """
    Finds local peaks in a signal using basic numpy operations (no scipy dependency).
    Respects minimum spacing between peaks and a height threshold.
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
    Estimates the cardiac cycle period from autocorrelation of the late-time signal.
    Looks for the dominant frequency within the specified period range (typically 0.4-1.5 seconds).
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


def extract_last_cycle_by_period(signal_data: np.ndarray, 
                                  time_data: np.ndarray,
                                  period: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extracts the last complete cardiac cycle based on detected peaks in the signal.
    Uses the signal peaks to identify cycle boundaries rather than relying solely on the period.
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


def extract_n_cycles(signal_data: np.ndarray, 
                     time_data: np.ndarray,
                     n_cycles: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts the last n complete cardiac cycles from the signal for overlay comparison.
    Useful for assessing cycle-to-cycle variability and convergence.
    """
    if signal_data is None or len(signal_data) < 20:
        return []
    
    prominence_threshold = np.max(signal_data) * 0.05
    peaks = numpy_find_peaks(signal_data, distance=5, height=prominence_threshold)
    
    if len(peaks) < n_cycles + 1:
        return []
    
    cycles = []
    for i in range(n_cycles):
        start_idx = peaks[-(n_cycles + 1) + i]
        end_idx = peaks[-(n_cycles) + i]
        end_idx = min(end_idx + 1, len(signal_data))
        
        cycle_signal = signal_data[start_idx:end_idx]
        cycle_time = time_data[start_idx:end_idx]
        cycles.append((cycle_signal, cycle_time))
    
    return cycles


def extract_last_cycle_time_window(signal_data: np.ndarray,
                                   time_data: np.ndarray,
                                   period: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extracts the last period-length window of data from the signal.
    Simple time-based slicing that doesn't depend on peak detection.
    Returns the cycle with time values shifted to start at zero.
    """
    if signal_data is None or len(signal_data) < 10 or period is None or period <= 0:
        return None, None
    
    t_end = time_data[-1]
    t_start = t_end - period
    
    mask = (time_data >= t_start) & (time_data <= t_end)
    indices = np.where(mask)[0]
    
    if len(indices) < 10:
        return None, None
    
    cycle_signal = signal_data[indices]
    cycle_time = time_data[indices]
    cycle_time_relative = cycle_time - cycle_time[0]
    
    return cycle_signal, cycle_time_relative


def load_global_metrics(metrics_file: Path) -> Optional[Dict[str, float]]:
    """
    Reads the global metrics CSV file from biological_analysis if it exists.
    Returns a dictionary of all hemodynamic metrics, or None if the file cannot be read.
    """
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            
            metrics = {}
            for key, val in row.items():
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = None
            
            return metrics
    except Exception:
        return None


def make_aortic_pressure_last_cycle(results_dir: Path, output_dir: Path, period: Optional[float] = None):
    """
    Creates Figure 1: Shows the aortic pressure waveform during the last cardiac cycle.
    """
    aorta_file = results_dir / 'heart_kim_lit' / 'aorta.txt'
    
    if not aorta_file.exists():
        print("Warning: aorta.txt not found, skipping fig1")
        return
    
    data = load_timeseries(aorta_file)
    if data is None:
        print("Warning: Could not load aorta.txt, skipping fig1")
        return
    
    time_col_idx = detect_time_column(data)
    if time_col_idx is None:
        print("Warning: Could not detect time column in aorta.txt, skipping fig1")
        return
    
    pressure_col_idx = select_pulsatile_column(data, time_col_idx)
    if pressure_col_idx is None:
        print("Warning: Could not detect pulsatile column in aorta.txt, skipping fig1")
        return
    
    time_col = data[:, time_col_idx]
    pressure_col = data[:, pressure_col_idx]
    
    pressure_col = (pressure_col - PRESSURE_BASELINE) / 133.322
    
    if period is None:
        period = estimate_period_autocorr(pressure_col, time_col)
    
    p_cycle, t_cycle = extract_last_cycle_by_period(pressure_col, time_col, period)
    
    if p_cycle is None:
        print("Warning: Could not extract cycle from pressure, skipping fig1")
        return
    
    t_cycle_relative = t_cycle - t_cycle[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_cycle_relative, p_cycle, 'b-', linewidth=2.5)
    ax.fill_between(t_cycle_relative, p_cycle, alpha=0.2)
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pressure (mmHg)', fontsize=14, fontweight='bold')
    ax.set_title('Aortic Pressure - Last Cardiac Cycle', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    
    fig.tight_layout()
    output_file = output_dir / 'fig1_aortic_pressure_last_cycle.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [1] fig1_aortic_pressure_last_cycle.png")


def make_aortic_pressure_cycle_overlay(results_dir: Path, output_dir: Path, period: Optional[float] = None):
    """
    Creates Figure 2: Overlays multiple cardiac cycles to show pressure waveform repeatability.
    """
    aorta_file = results_dir / 'heart_kim_lit' / 'aorta.txt'
    
    if not aorta_file.exists():
        print("Warning: aorta.txt not found, skipping fig2")
        return
    
    data = load_timeseries(aorta_file)
    if data is None:
        print("Warning: Could not load aorta.txt, skipping fig2")
        return
    
    time_col_idx = detect_time_column(data)
    if time_col_idx is None:
        print("Warning: Could not detect time column in aorta.txt, skipping fig2")
        return
    
    pressure_col_idx = select_pulsatile_column(data, time_col_idx)
    if pressure_col_idx is None:
        print("Warning: Could not detect pulsatile column in aorta.txt, skipping fig2")
        return
    
    time_col = data[:, time_col_idx]
    pressure_col = data[:, pressure_col_idx]
    
    pressure_col = (pressure_col - PRESSURE_BASELINE) / 133.322
    
    cycles = extract_n_cycles(pressure_col, time_col, n_cycles=4)
    
    if len(cycles) < 2:
        print("Warning: Could not extract multiple cycles, skipping fig2")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (cycle_signal, cycle_time) in enumerate(cycles):
        t_norm = (cycle_time - cycle_time[0]) / (cycle_time[-1] - cycle_time[0] + 1e-16)
        label = 'Cycle {}'.format(len(cycles) - i)
        ax.plot(t_norm, cycle_signal, '-', linewidth=2, 
                color=colors[i % len(colors)], label=label, alpha=0.8)
    
    ax.set_xlabel('Normalized Time (cycle fraction)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pressure (mmHg)', fontsize=14, fontweight='bold')
    ax.set_title('Aortic Pressure - Cycle Overlay', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    
    fig.tight_layout()
    output_file = output_dir / 'fig2_aortic_pressure_cycle_overlay.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [2] fig2_aortic_pressure_cycle_overlay.png")


def make_aortic_flow_last_cycle(results_dir: Path, output_dir: Path, period: Optional[float] = None):
    """
    Creates Figure 3: Displays aortic flow rate during the last cardiac cycle.
    Shows both forward and reverse flow patterns.
    """
    flow_file = results_dir / 'heart_kim_lit' / 'L_lv_aorta.txt'
    
    if not flow_file.exists():
        print("Warning: L_lv_aorta.txt not found, skipping fig3")
        return
    
    data = load_timeseries(flow_file)
    if data is None:
        print("Warning: Could not load L_lv_aorta.txt, skipping fig3")
        return
    
    time_col_idx = detect_time_column(data)
    if time_col_idx is None:
        print("Warning: Could not detect time column in L_lv_aorta.txt, skipping fig3")
        return
    
    flow_col_idx = select_pulsatile_column(data, time_col_idx)
    if flow_col_idx is None:
        print("Warning: Could not detect pulsatile column in L_lv_aorta.txt, skipping fig3")
        return
    
    time_col = data[:, time_col_idx]
    flow_col = data[:, flow_col_idx]
    
    flow_col_ml_s = flow_col * 1e6
    
    if period is None:
        aorta_file = results_dir / 'heart_kim_lit' / 'aorta.txt'
        if aorta_file.exists():
            aorta_data = load_timeseries(aorta_file)
            if aorta_data is not None:
                aorta_time_idx = detect_time_column(aorta_data)
                aorta_press_idx = select_pulsatile_column(aorta_data, aorta_time_idx)
                if aorta_time_idx is not None and aorta_press_idx is not None:
                    period = estimate_period_autocorr(
                        aorta_data[:, aorta_press_idx],
                        aorta_data[:, aorta_time_idx]
                    )
        
        if period is None:
            period = estimate_period_autocorr(flow_col, time_col)
    
    f_cycle, t_cycle_relative = extract_last_cycle_time_window(flow_col_ml_s, time_col, period)
    
    if f_cycle is None:
        print("Warning: Could not extract cycle from flow, skipping fig3")
        return
    
    if period is not None and t_cycle_relative[-1] < 0.5 * period:
        print("Warning: Flow cycle window too short; check period detection")
    
    mean_flow = np.mean(f_cycle)
    peak_flow = np.max(f_cycle)
    cycle_duration = t_cycle_relative[-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_cycle_relative, f_cycle, 'r-', linewidth=2.5)
    ax.fill_between(t_cycle_relative, f_cycle, alpha=0.2, color='red')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Flow (mL/s)', fontsize=14, fontweight='bold')
    ax.set_title('Aortic Flow - Last Cardiac Cycle', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    
    fig.tight_layout()
    output_file = output_dir / 'fig3_aortic_flow_last_cycle.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [3] fig3_aortic_flow_last_cycle.png")
    print("      Mean flow: {:.1f} mL/s, Peak flow: {:.1f} mL/s, Duration: {:.3f} s".format(
        mean_flow, peak_flow, cycle_duration))


def make_summary_metrics_figure(metrics: Dict[str, float], output_dir: Path):
    """
    Creates Figure 4: A formatted table of key hemodynamic metrics for the simulation.
    """
    if metrics is None:
        print("Warning: No metrics available, skipping fig4")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    metric_labels = {
        'HR_bpm': 'Heart Rate',
        'cycle_period_s': 'Cycle Period',
        'P_sys_mmHg': 'Systolic Pressure',
        'P_dia_mmHg': 'Diastolic Pressure',
        'P_map_mmHg': 'Mean Arterial Pressure',
        'SV_ml': 'Stroke Volume',
        'CO_lmin': 'Cardiac Output',
        'rms_convergence_pct': 'Convergence RMS%',
    }
    
    metric_units = {
        'HR_bpm': 'bpm',
        'cycle_period_s': 's',
        'P_sys_mmHg': 'mmHg',
        'P_dia_mmHg': 'mmHg',
        'P_map_mmHg': 'mmHg',
        'SV_ml': 'mL',
        'CO_lmin': 'L/min',
        'rms_convergence_pct': '%',
    }
    
    metric_formats = {
        'HR_bpm': '{:.1f}',
        'cycle_period_s': '{:.4f}',
        'P_sys_mmHg': '{:.1f}',
        'P_dia_mmHg': '{:.1f}',
        'P_map_mmHg': '{:.1f}',
        'SV_ml': '{:.1f}',
        'CO_lmin': '{:.2f}',
        'rms_convergence_pct': '{:.4f}',
    }
    
    table_data = []
    for key in metric_labels.keys():
        if key in metrics and metrics[key] is not None:
            label = metric_labels[key]
            value = metrics[key]
            unit = metric_units[key]
            fmt = metric_formats[key]
            value_str = fmt.format(value) + ' ' + unit
            table_data.append([label, value_str])
    
    if not table_data:
        print("Warning: No valid metrics to display, skipping fig4")
        return
    
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 3)
    
    for i in range(len(table_data) + 1):
        cell = table[(i, 0)]
        cell.set_facecolor('#e6f2ff' if i % 2 == 0 else '#ffffff')
        cell.set_text_props(weight='bold' if i == 0 else 'normal')
        
        cell = table[(i, 1)]
        cell.set_facecolor('#e6f2ff' if i % 2 == 0 else '#ffffff')
        cell.set_text_props(weight='bold' if i == 0 else 'normal')
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    ax.set_title('FirstBlood Hemodynamic Metrics Summary', 
                 fontsize=16, fontweight='bold', pad=20)
    
    fig.tight_layout()
    output_file = output_dir / 'fig4_summary_metrics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [4] fig4_summary_metrics.png")


def read_csv_column(filepath: Path, column_name: str) -> Optional[List[str]]:
    """
    Extracts a single column from a CSV file. Returns None if the file doesn't exist or the column isn't found.
    """
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or column_name not in reader.fieldnames:
                return None
            
            values = []
            for row in reader:
                values.append(row[column_name])
            return values
    except Exception:
        return None


def read_csv_dict(filepath: Path) -> Optional[List[Dict[str, str]]]:
    """
    Reads an entire CSV file with headers into a list of dictionaries (one dict per row).
    Returns None if the file cannot be read.
    """
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows if rows else None
    except Exception:
        return None


def find_signal_file(results_dir: Path, arterial_name: str) -> Optional[Path]:
    """
    Searches for an arterial signal file by name (case-insensitive).
    Looks in both the results directory and its arterial/ subdirectory.
    Returns the full path if found, None otherwise.
    """
    search_dirs = [results_dir, results_dir / 'arterial']
    search_terms = [
        arterial_name.lower(),
        arterial_name.replace(' ', '_').lower(),
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for txt_file in search_dir.glob('*.txt'):
            file_lower = txt_file.name.lower()
            for term in search_terms:
                if term in file_lower:
                    return txt_file
    
    return None


def compute_graph_distance(edges_list: List[Tuple[str, str]], root_node: str) -> Dict[str, int]:
    """
    Uses breadth-first search to compute distances from a root node to all connected nodes.
    Useful for determining hierarchical levels in the vascular tree.
    """
    neighbors = defaultdict(list)
    all_nodes = set()
    
    for node1, node2 in edges_list:
        neighbors[node1].append(node2)
        neighbors[node2].append(node1)
        all_nodes.add(node1)
        all_nodes.add(node2)
    
    distances = {node: float('inf') for node in all_nodes}
    distances[root_node] = 0
    
    queue = deque([root_node])
    while queue:
        node = queue.popleft()
        for neighbor in neighbors[node]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    
    return distances


def make_paper_style_arterial_tree(results_dir: Path, output_dir: Path, model_name: str):
    """
    Creates Figure 6: A detailed anatomical diagram of the full 55-vessel arterial network.
    Vessels are colored by flow direction: red for forward flow, blue for reversed flow,
    and gray for minimal flow. Layout follows anatomical organization: brain at top, thorax in middle, legs at bottom.
    """
    M3S_TO_MLMIN = 60.0 * 1e6
    eps = 0.01
    
    def load_mean_flow_robust(vessel_id):
        """Reads mean flow data for a vessel and converts to mL/min regardless of file format."""
        file_path = results_dir / "arterial" / f"{vessel_id}.txt"
        if not file_path.exists():
            return 0.0
        try:
            data = load_timeseries(file_path)
            if data is None or data.shape[0] < 2:
                return 0.0
            
            time_idx = detect_time_column(data)
            if time_idx is None:
                time_idx = 0
            
            flow_idx = select_pulsatile_column(data, time_idx)
            if flow_idx is None:
                if data.shape[1] >= 2:
                    flow_idx = 1
                else:
                    return 0.0
            
            flow_col = data[:, flow_idx]
            mean_flow_m3s = np.mean(flow_col)
            return mean_flow_m3s * M3S_TO_MLMIN
        except:
            return 0.0
    
    def vessel_color(flow):
        """Returns color and line width based on flow direction: blue for reversed, red for forward, gray for minimal."""
        if flow < -eps:
            return 'blue', 2.2
        elif abs(flow) <= eps:
            return '#bbbbbb', 0.9
        else:
            return 'red', 1.6
    
    # Vessel IDs are mapped to anatomical names for labeling.
    vessel_names = {
        'A1': 'Aorta', 'A2': 'L-Carotid', 'A3': 'R-Carotid', 'A4': 'R-Subclavian',
        'A5': 'R-CCA', 'A6': 'Vertebral', 'A12': 'R-ICA', 'A13': 'R-ECA', 
        'A14': 'L-CCA', 'A15': 'L-CCA', 'A16': 'L-ICA', 'A17': 'L-ECA',
        'A20': 'L-Vertebral', 'A27': 'Coeliac', 'A28': 'SMA', 'A35': 'Renal-L',
        'A42': 'R-Iliac', 'A43': 'L-Iliac', 'A44': 'R-Femoral', 'A50': 'L-Femoral',
        'A56': 'Basilar', 'A59': 'Basilar-Bifurcation', 'A60': 'R-PCA', 'A61': 'L-PCA',
        'A62': 'R-PCom', 'A63': 'L-PCom', 'A68': 'R-ACA', 'A69': 'L-ACA',
        'A70': 'R-MCA', 'A71': 'R-MCA-M2a', 'A73': 'L-MCA', 'A79': 'R-ICA-siphon',
        'A81': 'L-ICA-siphon', 'A101': 'R-ACA-A1', 'A103': 'L-ACA-A1',
    }
    
    def draw_vessel(ax, pts, vid, label=None, label_pos=0.5):
        """Draws a vessel as a line connecting the given points. Loads flow data to color-code the vessel.
        Optionally adds an anatomical label along the vessel path."""
        flow = load_mean_flow_robust(vid)
        color, lw = vessel_color(flow)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle='round',
                zorder=2, alpha=0.85)
        
        # The provided label is used when available; otherwise the name map is used.
        if label is None and vid in vessel_names:
            label = vessel_names[vid]
        
        if label:
            # The label is placed along the vessel at the requested fractional position.
            idx = max(0, min(int(len(xs) * label_pos), len(xs) - 1))
            x_label, y_label = xs[idx], ys[idx]
            ax.text(x_label, y_label, label, fontsize=5.5, fontweight='bold',
                   ha='center', va='bottom', zorder=5,
                   bbox=dict(boxstyle='round,pad=0.15', fc='white', 
                            ec='none', alpha=0.75))
        
        return flow
    
    def junc(ax, x, y):
        """Marks a junction point where vessels meet (white circle with black outline)."""
        ax.plot(x, y, 'o', color='white', markersize=6,
                markeredgecolor='black', markeredgewidth=0.9, zorder=4)
    
    def term(ax, x, y):
        """Marks the end of a terminal vessel (gray filled circle)."""
        ax.plot(x, y, 'o', color='#777777', markersize=4.5, zorder=4)
    
    fig, ax = plt.subplots(figsize=(9, 17))
    ax.set_xlim(-9, 9)
    ax.set_ylim(-1, 39)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.suptitle(f'Arterial Network: {model_name}',
                 fontsize=14, fontweight='bold', y=0.99)
    
    # Vertical region labels are placed along the left side of the diagram.
    ax.text(-8.2, 37.0, 'BRAIN', fontsize=11, fontweight='bold', 
           color='#333333', rotation=90, va='center',
           bbox=dict(boxstyle='round,pad=0.5', fc='#ffffcc', ec='#999999', linewidth=1.5, alpha=0.9))
    ax.text(-8.2, 19.5, 'TORSO', fontsize=11, fontweight='bold',
           color='#333333', rotation=90, va='center',
           bbox=dict(boxstyle='round,pad=0.5', fc='#ffcccc', ec='#999999', linewidth=1.5, alpha=0.9))
    ax.text(-8.2, 6.5, 'LEGS', fontsize=11, fontweight='bold',
           color='#333333', rotation=90, va='center',
           bbox=dict(boxstyle='round,pad=0.5', fc='#ccccff', ec='#999999', linewidth=1.5, alpha=0.9))
    
    # Brain vasculature is laid out here: carotids, vertebrals, and the Circle of Willis (y ≈ 26-38).
    r_cca_top = (1.2, 27.0)
    l_cca_top = (-1.2, 27.0)
    r_ica = (1.0, 28.5)
    l_ica = (-1.0, 28.5)
    r_eca = (2.2, 27.5)
    l_eca = (-2.2, 27.5)
    r_pcoa_jxn = (0.8, 29.5)
    l_pcoa_jxn = (-0.8, 29.5)
    r_m1 = (0.7, 30.5)
    l_m1 = (-0.7, 30.5)
    ba_bif = (0, 29.0)
    ba_mid = (0, 28.2)
    vert_conf = (0, 27.2)
    r_p1 = (0.5, 29.8)
    l_p1 = (-0.5, 29.8)
    r_mca_end = (4.0, 31.5)
    r_mca_m2a = (4.5, 32.5)
    r_mca_m2b = (3.5, 33.0)
    r_aca_a1 = (1.5, 31.5)
    r_aca_a2 = (2.0, 33.0)
    r_pca_p2 = (2.5, 30.5)
    l_mca_end = (-4.0, 31.5)
    l_mca_m2a = (-4.5, 32.5)
    l_mca_m2b = (-3.5, 33.0)
    l_aca_a1 = (-1.5, 31.5)
    l_aca_a2 = (-2.0, 33.0)
    l_pca_p2 = (-2.5, 30.5)
    
    draw_vessel(ax, [r_cca_top, r_ica], 'A12', label='R-ICA')
    draw_vessel(ax, [r_cca_top, r_eca], 'A13')
    junc(ax, *r_cca_top)
    junc(ax, *r_ica)
    
    draw_vessel(ax, [l_cca_top, l_ica], 'A16', label='L-ICA')
    draw_vessel(ax, [l_cca_top, l_eca], 'A17')
    junc(ax, *l_cca_top)
    junc(ax, *l_ica)
    
    r_eca2 = (3.0, 27.0)
    r_eca3 = (3.5, 28.0)
    r_eca4 = (4.0, 27.5)
    draw_vessel(ax, [r_eca, r_eca2], 'A83')
    junc(ax, *r_eca)
    junc(ax, *r_eca2)
    draw_vessel(ax, [r_eca2, r_eca3], 'A87')
    junc(ax, *r_eca3)
    draw_vessel(ax, [r_eca3, r_eca4], 'A91')
    term(ax, *r_eca4)
    draw_vessel(ax, [r_eca3, (4.5, 27.2)], 'A92')
    term(ax, 4.5, 27.2)
    draw_vessel(ax, [r_eca2, (3.8, 26.2)], 'A88')
    term(ax, 3.8, 26.2)
    draw_vessel(ax, [r_eca, (2.8, 26.5)], 'A84')
    term(ax, 2.8, 26.5)
    
    l_eca2 = (-3.0, 27.0)
    l_eca3 = (-3.5, 28.0)
    l_eca4 = (-4.0, 27.5)
    draw_vessel(ax, [l_eca, l_eca2], 'A85')
    junc(ax, *l_eca)
    junc(ax, *l_eca2)
    draw_vessel(ax, [l_eca2, l_eca3], 'A89')
    junc(ax, *l_eca3)
    draw_vessel(ax, [l_eca3, l_eca4], 'A93')
    term(ax, *l_eca4)
    draw_vessel(ax, [l_eca3, (-4.5, 27.2)], 'A94')
    term(ax, -4.5, 27.2)
    draw_vessel(ax, [l_eca2, (-3.8, 26.2)], 'A90')
    term(ax, -3.8, 26.2)
    draw_vessel(ax, [l_eca, (-2.8, 26.5)], 'A86')
    term(ax, -2.8, 26.5)
    
    draw_vessel(ax, [r_ica, r_pcoa_jxn], 'A79')
    junc(ax, *r_pcoa_jxn)
    draw_vessel(ax, [r_pcoa_jxn, (0.75, 30.0)], 'A66')
    junc(ax, 0.75, 30.0)
    draw_vessel(ax, [(0.75, 30.0), r_m1], 'A101')
    junc(ax, *r_m1)
    
    draw_vessel(ax, [l_ica, l_pcoa_jxn], 'A81')
    junc(ax, *l_pcoa_jxn)
    draw_vessel(ax, [l_pcoa_jxn, (-0.75, 30.0)], 'A67')
    junc(ax, -0.75, 30.0)
    draw_vessel(ax, [(-0.75, 30.0), l_m1], 'A103')
    junc(ax, *l_m1)
    
    draw_vessel(ax, [r_ica, (1.7, 28.8)], 'A80')
    term(ax, 1.7, 28.8)
    draw_vessel(ax, [l_ica, (-1.7, 28.8)], 'A82')
    term(ax, -1.7, 28.8)
    
    junc(ax, *vert_conf)
    draw_vessel(ax, [vert_conf, ba_mid], 'A56')
    junc(ax, *ba_mid)
    draw_vessel(ax, [ba_mid, (0.6, 28.0)], 'A57')
    term(ax, 0.6, 28.0)
    draw_vessel(ax, [ba_mid, (-0.6, 28.0)], 'A58')
    term(ax, -0.6, 28.0)
    
    draw_vessel(ax, [ba_mid, ba_bif], 'A59', label='Basilar')
    junc(ax, *ba_bif)
    
    r_pca_flow = draw_vessel(ax, [ba_bif, r_p1], 'A60', label='R-PCA')
    junc(ax, *r_p1)
    draw_vessel(ax, [ba_bif, l_p1], 'A61', label='L-PCA')
    junc(ax, *l_p1)
    
    draw_vessel(ax, [r_p1, r_pcoa_jxn], 'A62')
    draw_vessel(ax, [l_p1, l_pcoa_jxn], 'A63')
    
    draw_vessel(ax, [r_p1, r_pca_p2], 'A64')
    term(ax, *r_pca_p2)
    draw_vessel(ax, [l_p1, l_pca_p2], 'A65')
    term(ax, *l_pca_p2)
    
    draw_vessel(ax, [r_m1, r_mca_end], 'A70', label='R-MCA')
    junc(ax, *r_mca_end)
    draw_vessel(ax, [r_mca_end, r_mca_m2a], 'A71')
    term(ax, *r_mca_m2a)
    draw_vessel(ax, [r_mca_end, r_mca_m2b], 'A72')
    term(ax, *r_mca_m2b)
    
    draw_vessel(ax, [l_m1, l_mca_end], 'A73', label='L-MCA')
    junc(ax, *l_mca_end)
    draw_vessel(ax, [l_mca_end, l_mca_m2a], 'A74')
    term(ax, *l_mca_m2a)
    draw_vessel(ax, [l_mca_end, l_mca_m2b], 'A75')
    term(ax, *l_mca_m2b)
    
    draw_vessel(ax, [r_m1, r_aca_a1], 'A68', label='R-ACA')
    junc(ax, *r_aca_a1)
    draw_vessel(ax, [r_aca_a1, r_aca_a2], 'A76')
    term(ax, *r_aca_a2)
    
    draw_vessel(ax, [l_m1, l_aca_a1], 'A69', label='L-ACA')
    junc(ax, *l_aca_a1)
    draw_vessel(ax, [l_aca_a1, l_aca_a2], 'A78')
    term(ax, *l_aca_a2)
    
    draw_vessel(ax, [l_aca_a1, r_aca_a1], 'A77')
    
    draw_vessel(ax, [(0.75, 30.0), (1.3, 30.8)], 'A100')
    term(ax, 1.3, 30.8)
    draw_vessel(ax, [(-0.75, 30.0), (-1.3, 30.8)], 'A102')
    term(ax, -1.3, 30.8)
    
    # Reversed PCA flow is flagged here (fetal pattern).
    if r_pca_flow < 0:
        ax.annotate('FLOW REVERSAL\n(Fetal variant)',
                    xy=(0.5, 29.8), xytext=(3.0, 29.0),
                    fontsize=7, fontweight='bold', color='blue',
                    bbox=dict(boxstyle='round', fc='cyan', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    zorder=6)
    
    # Thoracic and abdominal aorta branches are laid out here (y ≈ 13-26).
    # The cardiac inlet marks the boundary where blood enters from the left ventricle.
    heart_inlet = (0, 24.0)
    ax.plot(*heart_inlet, marker='D', markerfacecolor='red', markeredgecolor='darkred',
            markersize=10, markeredgewidth=1.5, zorder=10)
    ax.text(0.9, 24.0, 'HEART', fontsize=8, fontweight='bold', 
            color='darkred', va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='mistyrose', 
                     ec='darkred', lw=1, alpha=0.9))
    
    # The ascending aorta (A1) starts at the heart.
    ax.text(-1.3, 24.8, 'Asc. Aorta', fontsize=7, fontweight='bold',
            style='italic', color='darkred',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', 
                     ec='red', lw=1, alpha=0.85))
    
    n1 = (0, 25.2)
    n2 = (0, 25.8)
    draw_vessel(ax, [heart_inlet, n1], 'A1', label='Aorta', label_pos=0.3)
    junc(ax, *n1)
    draw_vessel(ax, [n1, (-0.6, 24.8)], 'A96')
    term(ax, -0.6, 24.8)
    draw_vessel(ax, [n1, (-0.8, 25.0)], 'A97')
    junc(ax, -0.8, 25.0)
    draw_vessel(ax, [(-0.8, 25.0), (-1.2, 24.6)], 'A98')
    term(ax, -1.2, 24.6)
    draw_vessel(ax, [(-0.8, 25.0), (-1.0, 25.2)], 'A99')
    term(ax, -1.0, 25.2)
    
    draw_vessel(ax, [n1, n2], 'A95')
    junc(ax, *n2)
    
    n6 = (1.5, 25.5)
    draw_vessel(ax, [n2, n6], 'A3')
    junc(ax, *n6)
    
    n3 = (-1.0, 25.8)
    draw_vessel(ax, [n2, n3], 'A2')
    junc(ax, *n3)
    
    n4 = (-2.0, 25.5)
    draw_vessel(ax, [n3, n4], 'A14')
    junc(ax, *n4)
    
    r_cca_bot = (1.2, 26.2)
    draw_vessel(ax, [n6, r_cca_bot], 'A5', label='R-CCA')
    junc(ax, *r_cca_bot)
    ax.plot([r_cca_bot[0], r_cca_top[0]], [r_cca_bot[1], r_cca_top[1]],
            color='red', linewidth=1.6, alpha=0.85, zorder=2)
    
    l_cca_bot = (-1.2, 26.2)
    draw_vessel(ax, [n3, l_cca_bot], 'A15', label='L-CCA')
    junc(ax, *l_cca_bot)
    ax.plot([l_cca_bot[0], l_cca_top[0]], [l_cca_bot[1], l_cca_top[1]],
            color='red', linewidth=1.6, alpha=0.85, zorder=2)
    
    n7 = (2.5, 24.8)
    draw_vessel(ax, [n6, n7], 'A4', label='R-Subclavian')
    junc(ax, *n7)
    draw_vessel(ax, [n7, (1.5, 26.0), vert_conf], 'A6', label='R-Vertebral')
    
    n10 = (-2.8, 24.8)
    draw_vessel(ax, [n4, n10], 'A19', label='L-Subclavian')
    junc(ax, *n10)
    draw_vessel(ax, [n10, (-1.5, 26.0), vert_conf], 'A20')
    
    n8 = (4.0, 23.0)
    draw_vessel(ax, [n7, (3.2, 24.2), n8], 'A7')
    junc(ax, *n8)
    
    n9 = (4.5, 21.5)
    draw_vessel(ax, [n8, n9], 'A9')
    junc(ax, *n9)
    
    draw_vessel(ax, [n8, (4.8, 22.0)], 'A8')
    term(ax, 4.8, 22.0)
    draw_vessel(ax, [n9, (5.0, 20.5)], 'A10')
    term(ax, 5.0, 20.5)
    draw_vessel(ax, [n9, (4.2, 20.2)], 'A11')
    term(ax, 4.2, 20.2)
    
    n11 = (-4.0, 23.0)
    draw_vessel(ax, [n10, (-3.2, 24.2), n11], 'A21')
    junc(ax, *n11)
    
    n12 = (-4.5, 21.5)
    draw_vessel(ax, [n11, n12], 'A23')
    junc(ax, *n12)
    
    draw_vessel(ax, [n11, (-4.8, 22.0)], 'A22')
    term(ax, -4.8, 22.0)
    draw_vessel(ax, [n12, (-5.0, 20.5)], 'A24')
    term(ax, -5.0, 20.5)
    draw_vessel(ax, [n12, (-4.2, 20.2)], 'A25')
    term(ax, -4.2, 20.2)
    
    n51 = (0, 23.5)
    draw_vessel(ax, [n4, (-1.5, 24.8), (-0.5, 24.0), n51], 'A18')
    junc(ax, *n51)
    draw_vessel(ax, [n51, (0.8, 23.3)], 'A26')
    term(ax, 0.8, 23.3)
    
    n52 = (0, 22.0)
    draw_vessel(ax, [n51, n52], 'A27')
    junc(ax, *n52)
    
    draw_vessel(ax, [n52, (-0.9, 21.8)], 'A29')
    junc(ax, -0.9, 21.8)
    draw_vessel(ax, [(-0.9, 21.8), (-1.5, 21.5)], 'A30')
    junc(ax, -1.5, 21.5)
    draw_vessel(ax, [(-1.5, 21.5), (-2.0, 21.2)], 'A31')
    term(ax, -2.0, 21.2)
    draw_vessel(ax, [(-1.5, 21.5), (-1.8, 20.8)], 'A33')
    term(ax, -1.8, 20.8)
    draw_vessel(ax, [(-0.9, 21.8), (-1.2, 21.0)], 'A32')
    term(ax, -1.2, 21.0)
    
    n22 = (0, 20.5)
    draw_vessel(ax, [n52, n22], 'A28')
    junc(ax, *n22)
    draw_vessel(ax, [n22, (-0.9, 20.3)], 'A34')
    term(ax, -0.9, 20.3)
    
    n23 = (0, 19.0)
    draw_vessel(ax, [n22, n23], 'A35')
    junc(ax, *n23)
    draw_vessel(ax, [n23, (0.9, 18.8)], 'A36')
    term(ax, 0.9, 18.8)
    draw_vessel(ax, [n23, (-0.9, 18.8)], 'A38')
    term(ax, -0.9, 18.8)
    
    n24 = (0, 17.5)
    draw_vessel(ax, [n23, n24], 'A37')
    junc(ax, *n24)
    
    n25 = (0, 16.0)
    draw_vessel(ax, [n24, n25], 'A39')
    junc(ax, *n25)
    draw_vessel(ax, [n25, (-0.9, 15.8)], 'A40')
    term(ax, -0.9, 15.8)
    
    n13 = (0, 14.5)
    draw_vessel(ax, [n25, n13], 'A41')
    junc(ax, *n13)
    
    # Lower extremity arteries are laid out here: iliac, femoral, and tibial branches (y ≈ 0-14).
    n17 = (1.5, 13.0)
    draw_vessel(ax, [n13, n17], 'A42', label='R-Iliac')
    junc(ax, *n17)
    draw_vessel(ax, [n17, (0.8, 12.5)], 'A45')
    term(ax, 0.8, 12.5)
    
    n18 = (2.0, 11.5)
    draw_vessel(ax, [n17, n18], 'A44', label='R-Femoral')
    junc(ax, *n18)
    draw_vessel(ax, [n18, (2.8, 11.2)], 'A47')
    term(ax, 2.8, 11.2)
    
    n19 = (2.2, 9.5)
    draw_vessel(ax, [n18, n19], 'A46', label='R-Tibial')
    junc(ax, *n19)
    
    draw_vessel(ax, [n19, (2.1, 6.0)], 'A48')
    term(ax, 2.1, 6.0)
    draw_vessel(ax, [n19, (2.8, 6.0)], 'A49')
    term(ax, 2.8, 6.0)
    
    n14 = (-1.5, 13.0)
    draw_vessel(ax, [n13, n14], 'A43', label='L-Iliac')
    junc(ax, *n14)
    draw_vessel(ax, [n14, (-0.8, 12.5)], 'A51')
    term(ax, -0.8, 12.5)
    
    n15 = (-2.0, 11.5)
    draw_vessel(ax, [n14, n15], 'A50', label='L-Femoral')
    junc(ax, *n15)
    draw_vessel(ax, [n15, (-2.8, 11.2)], 'A53')
    term(ax, -2.8, 11.2)
    
    n16 = (-2.2, 9.5)
    draw_vessel(ax, [n15, n16], 'A52', label='L-Tibial')
    junc(ax, *n16)
    
    draw_vessel(ax, [n16, (-2.1, 6.0)], 'A54')
    term(ax, -2.1, 6.0)
    draw_vessel(ax, [n16, (-2.8, 6.0)], 'A55')
    term(ax, -2.8, 6.0)
    
    # Anatomical labels are added using the helper function.
    add_anatomical_labels_to_tree(ax)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Forward flow'),
        Line2D([0], [0], color='blue', linewidth=2, label='Reversed flow'),
        Line2D([0], [0], color='#bbbbbb', linewidth=1, label='Minimal flow'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=7, linewidth=0, label='Junction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#777777',
               markersize=5, linewidth=0, label='Terminal'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=7, markeredgewidth=1, linewidth=0, label='Heart'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5,
              framealpha=0.95)
    
    plt.tight_layout()
    output_file = output_dir / 'fig5_arterial_tree_paper_style.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  [6] fig5_arterial_tree_paper_style.png")


def make_network_schematic(model_dir: Path, output_dir: Path):
    """
    Creates an optional network topology diagram with automatic layout.
    Reads connection information from arterial.csv to show vessel relationships.
    """
    # This function is not used in the main visualization pipeline yet.
    arterial_csv = model_dir / 'arterial.csv'
    
    if not arterial_csv.exists():
        print("Warning: arterial.csv not found, skipping network schematic")
        return
    
    rows = read_csv_dict(arterial_csv)
    if rows is None or len(rows) == 0:
        print("Warning: Could not read arterial.csv, skipping network schematic")
        return
    
    edges = []
    descriptions = {}
    
    try:
        for row in rows:
            start = row.get('start_node', '').strip()
            end = row.get('end_node', '').strip()
            if start and end:
                edges.append((start, end))
                desc = row.get('name', '')
                descriptions[(start, end)] = desc.lower()
    except Exception:
        print("Warning: Error parsing arterial.csv, skipping network schematic")
        return
    
    if not edges:
        print("Warning: No edges found in arterial.csv, skipping network schematic")
        return
    
    all_nodes = set()
    for node1, node2 in edges:
        all_nodes.add(node1)
        all_nodes.add(node2)
    
    root_node = max(all_nodes, key=lambda n: sum(1 for e in edges if e[0] == n or e[1] == n))
    distances = compute_graph_distance(edges, root_node)
    
    node_degrees = {node: 0 for node in all_nodes}
    for node1, node2 in edges:
        node_degrees[node1] += 1
        node_degrees[node2] += 1
    
    layers = {}
    for node, dist in distances.items():
        if dist not in layers:
            layers[dist] = []
        layers[dist].append(node)
    
    finite_distances = {node: dist for node, dist in distances.items() if dist != float('inf')}
    if not finite_distances:
        print("Warning: No finite distances found in graph, skipping network schematic")
        return
    
    max_dist = max(finite_distances.values())
    positions = {}
    for layer_idx, nodes_in_layer in layers.items():
        if layer_idx == float('inf'):
            continue
        y = (max_dist - layer_idx) * 2
        n_nodes = len(nodes_in_layer)
        for i, node in enumerate(sorted(nodes_in_layer)):
            x = (i - n_nodes / 2.0) * 3
            positions[node] = (x, y)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for node1, node2 in edges:
        if node1 not in positions or node2 not in positions:
            continue
        x1, y1 = positions[node1]
        x2, y2 = positions[node2]
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1.5, alpha=0.7)
    
    for node in all_nodes:
        if node not in positions:
            continue
        x, y = positions[node]
        degree = node_degrees[node]
        if degree == 1:
            ax.plot(x, y, 'ko', markersize=6, zorder=5)
        else:
            ax.plot(x, y, 'o', color='lightgray', markersize=8, zorder=4)
        
        ax.text(x, y - 0.35, node, fontsize=8, ha='center', va='top')
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-2, (max_dist + 1) * 2 + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Network Topology (Auto-Layout)', fontsize=16, fontweight='bold', pad=20)
    
    fig.tight_layout()
    output_file = output_dir / 'network_schematic_autolayout.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [optional] network_schematic_autolayout.png")




def make_multisite_waveform_panels(results_dir: Path, model_dir: Path, output_dir: Path, period: Optional[float] = None):
    """
    Creates Figure 8: Side-by-side pressure and flow waveforms at five key arterial locations.
    Helps visualize how the pulse wave propagates through the vascular tree.
    """
    locations = [
        ('Aorta', ['aorta'], 'heart_kim_lit'),
        ('Carotid', ['carotid'], 'arterial'),
        ('Radial', ['radial'], 'arterial'),
        ('Femoral', ['femoral'], 'arterial'),
        ('Anterior Tibial', ['tibial', 'anterior'], 'arterial'),
    ]
    
    location_data = {}
    
    for loc_name, search_terms, default_dir in locations:
        pres_file = None
        flow_file = None
        
        search_dir = results_dir / default_dir if default_dir else results_dir
        
        for term in search_terms:
            if pres_file is None:
                pres_file = find_signal_file(search_dir, term)
            if flow_file is None:
                flow_file = find_signal_file(search_dir, term)
            if pres_file and flow_file:
                break
        
        if loc_name == 'Aorta':
            pres_file = results_dir / 'heart_kim_lit' / 'aorta.txt'
            flow_file = results_dir / 'heart_kim_lit' / 'L_lv_aorta.txt'
        
        pres_data = None
        flow_data = None
        
        if pres_file and pres_file.exists():
            pres_data = load_timeseries(pres_file)
        if flow_file and flow_file.exists():
            flow_data = load_timeseries(flow_file)
        
        if pres_data is not None or flow_data is not None:
            location_data[loc_name] = {
                'pres_data': pres_data,
                'flow_data': flow_data,
            }
    
    if not location_data:
        print("Warning: No waveform data found, skipping fig7")
        return
    
    n_locations = len(location_data)
    fig, axes = plt.subplots(n_locations, 2, figsize=(14, 4 * n_locations))
    
    if n_locations == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (loc_name, data) in enumerate(location_data.items()):
        pres_data = data['pres_data']
        flow_data = data['flow_data']
        
        ax_pres = axes[row_idx, 0]
        ax_flow = axes[row_idx, 1]
        
        if pres_data is not None:
            time_idx = detect_time_column(pres_data)
            pres_idx = select_pulsatile_column(pres_data, time_idx)
            
            if time_idx is not None and pres_idx is not None:
                time_col = pres_data[:, time_idx]
                pres_col = pres_data[:, pres_idx]
                pres_col_mmhg = (pres_col - PRESSURE_BASELINE) / 133.322
                
                if period is not None:
                    p_cycle, t_cycle = extract_last_cycle_time_window(pres_col_mmhg, time_col, period)
                else:
                    p_cycle, t_cycle = None, None
                
                if p_cycle is not None:
                    ax_pres.plot(t_cycle, p_cycle, 'b-', linewidth=2)
                    ax_pres.fill_between(t_cycle, p_cycle, alpha=0.2)
                    ax_pres.set_ylabel('Pressure (mmHg)', fontsize=11, fontweight='bold')
                else:
                    ax_pres.text(0.5, 0.5, 'Data unavailable', ha='center', va='center',
                               transform=ax_pres.transAxes, fontsize=10, color='red')
        else:
            ax_pres.text(0.5, 0.5, 'Data unavailable', ha='center', va='center',
                       transform=ax_pres.transAxes, fontsize=10, color='red')
        
        if flow_data is not None:
            time_idx = detect_time_column(flow_data)
            flow_idx = select_pulsatile_column(flow_data, time_idx)
            
            if time_idx is not None and flow_idx is not None:
                time_col = flow_data[:, time_idx]
                flow_col = flow_data[:, flow_idx]
                flow_col_ml_s = flow_col * 1e6
                
                if period is not None:
                    f_cycle, t_cycle = extract_last_cycle_time_window(flow_col_ml_s, time_col, period)
                else:
                    f_cycle, t_cycle = None, None
                
                if f_cycle is not None:
                    ax_flow.plot(t_cycle, f_cycle, 'r-', linewidth=2)
                    ax_flow.fill_between(t_cycle, f_cycle, alpha=0.2, color='red')
                    ax_flow.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
                    ax_flow.set_ylabel('Flow (mL/s)', fontsize=11, fontweight='bold')
                else:
                    ax_flow.text(0.5, 0.5, 'Data unavailable', ha='center', va='center',
                               transform=ax_flow.transAxes, fontsize=10, color='red')
        else:
            ax_flow.text(0.5, 0.5, 'Data unavailable', ha='center', va='center',
                       transform=ax_flow.transAxes, fontsize=10, color='red')
        
        ax_pres.grid(True, alpha=0.3, linestyle='--')
        ax_flow.grid(True, alpha=0.3, linestyle='--')
        ax_pres.tick_params(labelsize=10)
        ax_flow.tick_params(labelsize=10)
        
        ax_pres.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax_flow.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        
        ax_pres.text(-0.45, 0.5, loc_name, transform=ax_pres.transAxes,
                    fontsize=12, fontweight='bold', ha='right', va='center',
                    rotation=90)
    
    axes[0, 0].set_title('Pressure', fontsize=13, fontweight='bold')
    axes[0, 1].set_title('Flow', fontsize=13, fontweight='bold')
    
    fig.suptitle('Multi-Site Waveform Panels', fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    output_file = output_dir / 'fig7_multisite_waveforms.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [8] fig7_multisite_waveforms.png")


# Circle of Willis visualization functions (priority 1-7) start here.

def load_cow_flow_dict(cerebral_flow_file: Path) -> Dict[str, float]:
    """
    Loads mean flow values (mL/min) from biological_analysis output.
    """
    flow_data: Dict[str, float] = {}
    try:
        with open(cerebral_flow_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return flow_data
        row = rows[0]
        if 'Vessel' in row and 'Mean_Flow_mL_min' in row:
            for r in rows:
                vessel = r.get('Vessel', '').strip()
                flow = r.get('Mean_Flow_mL_min', None)
                if vessel and flow:
                    try:
                        flow_data[vessel] = float(flow)
                    except ValueError:
                        pass
        else:
            for key in ['L_ICA', 'R_ICA', 'Basilar', 'ACoA', 'L_PCoA', 'R_PCoA']:
                if key in row and row[key] not in (None, ''):
                    try:
                        flow_data[key] = float(row[key])
                    except ValueError:
                        pass
    except Exception:
        return {}
    return flow_data


def format_cow_flow_label(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    if abs(value) < 1.0:
        return '{:.3f} mL/min'.format(value)
    return '{:.1f} mL/min'.format(value)


def make_cow_topology_diagram(results_dir: Path,
                              model_dir: Path,
                              output_dir: Path,
                              flow_data: Optional[Dict[str, float]] = None):
    """
    Creates Circle of Willis topology diagram with patient-specific anatomy.
    Shows vessel diameters and flow values with anatomically accurate layout.
    """
    # Vessel geometry data is loaded when available.
    modifications_log = model_dir / 'modifications_log.csv'
    cerebral_flow_file = output_dir.parent / 'biological_analysis' / 'cerebral_flow_summary.csv'
    
    if not modifications_log.exists() or not cerebral_flow_file.exists():
        print("Warning: modifications_log.csv or cerebral_flow_summary.csv not found, skipping CoW topology")
        return
    
    # Vessel dimensions are loaded from the modifications log.
    vessel_diameters = {}
    try:
        with open(modifications_log, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vessel_id = row.get('vessel_id', '').strip()
                diameter = row.get('diameter_mm', None)
                if vessel_id and diameter:
                    try:
                        vessel_diameters[vessel_id] = float(diameter)
                    except ValueError:
                        pass
    except Exception:
        print("Warning: Could not parse modifications_log.csv, skipping CoW topology")
        return
    
    # Flow data is loaded from the biological analysis output.
    if flow_data is None:
        flow_data = load_cow_flow_dict(cerebral_flow_file)
    if not flow_data:
        print("Warning: Could not parse cerebral_flow_summary.csv, skipping CoW topology")
        return

    print("[COW] Flow dict: {}".format(flow_data))
    
    # Vessel positions are defined using an anatomical layout.
    vessel_positions = {
        'Basilar': (0, 0),
        'R-P1': (1.5, 1), 'L-P1': (-1.5, 1),
        'R-PCoA': (2, 2), 'L-PCoA': (-2, 2),
        'R-ICA': (3, 2.5), 'L-ICA': (-3, 2.5),
        'R-A1': (1, 3.5), 'L-A1': (-1, 3.5), 'ACoA': (0, 4),
        'R-MCA': (4, 4), 'L-MCA': (-4, 4),
        'R-A2': (1.5, 5), 'L-A2': (-1.5, 5),
        'R-PCA': (2, 2.5), 'L-PCA': (-2, 2.5),
    }
    
    # Vessel connections are defined for the topology graph.
    connections = [
        ('Basilar', 'R-P1'), ('Basilar', 'L-P1'),
        ('R-P1', 'R-PCoA'), ('L-P1', 'L-PCoA'),
        ('R-ICA', 'R-PCoA'), ('L-ICA', 'L-PCoA'),
        ('R-ICA', 'R-A1'), ('L-ICA', 'L-A1'),
        ('R-A1', 'ACoA'), ('L-A1', 'ACoA'),
        ('R-A1', 'R-MCA'), ('L-A1', 'L-MCA'),
        ('R-A1', 'R-A2'), ('L-A1', 'L-A2'),
        ('R-PCoA', 'R-PCA'), ('L-PCoA', 'L-PCA'),
    ]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    
    flow_key_map = {
        'L-ICA': 'L_ICA',
        'R-ICA': 'R_ICA',
        'Basilar': 'Basilar',
        'ACoA': 'ACoA',
        'L-PCoA': 'L_PCoA',
        'R-PCoA': 'R_PCoA',
    }

    # Vessels are drawn with flow-dependent coloring.
    for vessel_a, vessel_b in connections:
        if vessel_a in vessel_positions and vessel_b in vessel_positions:
            x1, y1 = vessel_positions[vessel_a]
            x2, y2 = vessel_positions[vessel_b]
            
            flow_key = flow_key_map.get(vessel_a, vessel_a)
            flow = flow_data.get(flow_key, None)
            diameter = vessel_diameters.get(vessel_a, 2.0)
            
            # Color is chosen based on flow direction.
            if flow is None or (isinstance(flow, float) and np.isnan(flow)):
                color = '#cccccc'
            elif flow < -0.5:
                color = 'blue'
            elif flow < 0.5:
                color = '#cccccc'
            else:
                color = 'red'
            
            linewidth = max(0.5, diameter / 2.0)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, zorder=2, alpha=0.8)
    
    # Vessel labels are drawn at their positions.
    for vessel, (x, y) in vessel_positions.items():
        diameter = vessel_diameters.get(vessel, 2.0)
        flow_key = flow_key_map.get(vessel, vessel)
        flow = flow_data.get(flow_key, None)
        
        # A marker is drawn at the vessel node.
        ax.plot(x, y, 'o', color='white', markersize=8, markeredgecolor='black', 
                markeredgewidth=1.5, zorder=4)
        
        # A label with vessel info is added.
        label_text = f'{vessel}\n{diameter:.2f}mm\n{format_cow_flow_label(flow)}'
        ax.text(x, y - 0.6, label_text, fontsize=7, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8),
                zorder=5)
    
    # Orientation labels are added.
    ax.text(0, -1.5, 'POSTERIOR', fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='gray', alpha=0.85))
    ax.text(0, 6.5, 'ANTERIOR', fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='gray', alpha=0.85))
    ax.text(-5, 3, 'LEFT', fontsize=10, fontweight='bold', ha='center', rotation=90,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', ec='gray', alpha=0.85))
    ax.text(5, 3, 'RIGHT', fontsize=10, fontweight='bold', ha='center', rotation=90,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', ec='gray', alpha=0.85))
    
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 7)
    ax.set_title('Circle of Willis - Patient-Specific Anatomy', fontsize=14, fontweight='bold', pad=20)
    
    # The legend is added.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Forward flow'),
        Line2D([0], [0], color='blue', linewidth=2, label='Reversed flow'),
        Line2D([0], [0], color='#cccccc', linewidth=2, label='Minimal flow'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / 'fig_cow_topology.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  [COW] fig_cow_topology.png")


def make_cow_waveform_grid(results_dir: Path, output_dir: Path, period: Optional[float] = None):
    """
    Creates 3x3 grid of FLOW waveforms for major Circle of Willis vessels.
    Shows flow variation through the cerebral circulation, which is vessel-specific.
    
        NOTE: Vessel files contain columns from solver_moc_io.cpp:
            [t, p_start, p_end, v_start, v_end, q_start, q_end, m_start, m_end, A_start, A_end, c_start, c_end]
        We use volume flow rate (q_start/q_end) or compute Q = v * A if needed.
    """
    # The nine key CoW vessels are defined here.
    vessels = [
        ('A12', 'R-ICA'), ('A16', 'L-ICA'), ('A56', 'Basilar'),
        ('A70', 'R-MCA'), ('A73', 'L-MCA'), ('A77', 'ACoA'),
        ('A62', 'R-PCoA'), ('A63', 'L-PCoA'), ('A64', 'R-PCA'),
    ]
    
    arterial_dir = results_dir / 'arterial'
    
    if not arterial_dir.exists():
        print("Warning: arterial directory not found, skipping CoW waveform grid")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    all_data_found = False
    print("\n[DEBUG] CoW Waveform Grid - Using volume flow rate (columns 5/6) for vessel-specific hemodynamics:")
    
    for idx, (vessel_id, vessel_name) in enumerate(vessels):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        vessel_file = arterial_dir / f'{vessel_id}.txt'
        
        if vessel_file.exists():
            data = load_timeseries(vessel_file)
            if data is not None:
                time_idx = detect_time_column(data)
                
                if time_idx is not None and data.shape[1] >= 11:
                    time_col = data[:, time_idx]
                    # Volume flow (q_start/q_end) is preferred; otherwise compute Q = v * A.
                    if data.shape[1] > 6:
                        flow_m3_s = 0.5 * (data[:, 5] + data[:, 6])
                    else:
                        vel_m_s = 0.5 * (data[:, 3] + data[:, 4])
                        area_m2 = 0.5 * (data[:, 9] + data[:, 10])
                        flow_m3_s = vel_m_s * area_m2
                    
                    # The flow is scaled to mL/min for readability.
                    flow_ml_min = flow_m3_s * 6e7
                    
                    # Basic flow statistics are printed for debugging.
                    print(f"  {vessel_id:4s} ({vessel_name:8s}): mean={np.mean(flow_ml_min):+.2f} mL/min, min={np.min(flow_ml_min):+.2f}, max={np.max(flow_ml_min):+.2f}")
                    
                    # The last cycle is extracted.
                    if period is not None:
                        f_cycle, t_cycle = extract_last_cycle_time_window(flow_ml_min, time_col, period)
                        if f_cycle is not None:
                            ax.plot(t_cycle, f_cycle, 'b-', linewidth=2)
                            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
                            ax.fill_between(t_cycle, f_cycle, where=(f_cycle >= 0), alpha=0.2, color='red', label='Forward')
                            ax.fill_between(t_cycle, f_cycle, where=(f_cycle < 0), alpha=0.2, color='blue', label='Reversed')
                            all_data_found = True
                        else:
                            ax.text(0.5, 0.5, 'No cycle', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=10, color='red')
                    else:
                        ax.plot(time_col, flow_ml_min, 'b-', linewidth=2)
                        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
                        all_data_found = True
                else:
                    print(f"  {vessel_id:4s} ({vessel_name:8s}): Invalid data structure")
            else:
                print(f"  {vessel_id:4s} ({vessel_name:8s}): Could not load timeseries")
        else:
            print(f"  {vessel_id:4s} ({vessel_name:8s}): File not found: {vessel_file}")
        
        ax.set_title(vessel_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Flow (mL/min)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    if not all_data_found:
        print("Warning: Could not load flow data for CoW vessels, skipping waveform grid")
        plt.close(fig)
        return
    
    # The y-axis is shared across all subplots.
    all_axes = axes.flatten()
    y_lim = None
    for ax in all_axes:
        if len(ax.lines) > 0:
            ylim = ax.get_ylim()
            if y_lim is None:
                y_lim = ylim
            else:
                y_lim = (min(y_lim[0], ylim[0]), max(y_lim[1], ylim[1]))
    
    if y_lim is not None:
        for ax in all_axes:
            ax.set_ylim(y_lim)
    
    fig.suptitle('Circle of Willis Flow Waveforms', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'fig_cow_waveforms_grid.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [GRID] fig_cow_waveforms_grid.png")


def add_anatomical_labels_to_tree(ax):
    """
    HELPER FUNCTION: Adds regional anatomical labels to the arterial tree diagram.
    Called within make_paper_style_arterial_tree() before legend creation.
    
    Adds three main region labels (Circle of Willis, Thoracic Aorta, Lower Extremities)
    and key vessel annotations.
    """
    # Main regional labels are added here.
    ax.text(0, 36, 'CIRCLE OF WILLIS', ha='center', fontsize=11, fontweight='bold',
            color='navy', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', 
                                   ec='navy', lw=2, alpha=0.85))
    
    ax.text(0, 20, 'THORACIC AORTA', ha='center', fontsize=10, fontweight='bold',
            color='darkred', bbox=dict(boxstyle='round,pad=0.4', fc='mistyrose',
                                      ec='darkred', lw=1.5, alpha=0.85))
    
    ax.text(0, 8, 'LOWER EXTREMITIES', ha='center', fontsize=10, fontweight='bold',
            color='darkgreen', bbox=dict(boxstyle='round,pad=0.4', fc='lightgreen',
                                        ec='darkgreen', lw=1.5, alpha=0.85))
    
    # Key vessel annotations are added here.
    key_vessels = [
        (1.0, 28.8, 'R-ICA'),
        (-1.0, 28.8, 'L-ICA'),
        (0.2, 28.5, 'Basilar'),
        (-1.4, 24.7, 'Asc. Aorta'),
    ]
    
    for x, y, label in key_vessels:
        ax.text(x, y, label, fontsize=7, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.7),
                zorder=10)


def make_patient_template_comparison(patient_dir: Path, template_dir: Path, output_dir: Path):
    """
    Creates side-by-side comparison of patient vs reference model (Abel_ref2) CoW flows.
    Shows differences in cerebral hemodynamics between patient and template.
    """
    # Patient flow data is loaded.
    patient_flow_file = patient_dir / 'cerebral_flow_summary.csv'
    template_flow_file = template_dir / 'cerebral_flow_summary.csv'
    
    if not patient_flow_file.exists():
        print("Warning: Patient cerebral_flow_summary.csv not found, skipping comparison")
        return
    
    if not template_flow_file.exists():
        print("Warning: Template cerebral_flow_summary.csv not found, skipping comparison")
        return
    
    # Data is loaded into patient and template dictionaries.
    patient_flows = {}
    template_flows = {}
    
    try:
        with open(patient_flow_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vessel = row.get('Vessel', '').strip()
                flow = row.get('Mean_Flow_mL_min', None)
                if vessel and flow:
                    try:
                        patient_flows[vessel] = float(flow)
                    except ValueError:
                        pass
    except Exception:
        print("Warning: Could not parse patient flow file, skipping comparison")
        return
    
    try:
        with open(template_flow_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vessel = row.get('Vessel', '').strip()
                flow = row.get('Mean_Flow_mL_min', None)
                if vessel and flow:
                    try:
                        template_flows[vessel] = float(flow)
                    except ValueError:
                        pass
    except Exception:
        print("Warning: Could not parse template flow file, skipping comparison")
        return
    
    # CoW vessels are defined in display order.
    vessels_order = [
        'R-ICA', 'L-ICA', 'Basilar', 'R-MCA', 'L-MCA',
        'R-ACA', 'L-ACA', 'R-PCA', 'L-PCA', 'ACoA',
        'R-PCoA', 'L-PCoA'
    ]
    
    # Flow values are extracted in display order.
    patient_data = [patient_flows.get(v, 0) for v in vessels_order]
    template_data = [template_flows.get(v, 0) for v in vessels_order]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(vessels_order))
    width = 0.6
    
    # The patient subplot is drawn.
    colors_p = ['red' if f > 0 else 'blue' if f < 0 else 'gray' for f in patient_data]
    ax1.barh(x, patient_data, width, color=colors_p, edgecolor='black', linewidth=0.7)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax1.set_yticks(x)
    ax1.set_yticklabels(vessels_order, fontsize=10)
    ax1.set_xlabel('Flow (mL/min)', fontsize=11, fontweight='bold')
    ax1.set_title('Patient CoW Flows', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # The template subplot is drawn.
    colors_t = ['red' if f > 0 else 'blue' if f < 0 else 'gray' for f in template_data]
    ax2.barh(x, template_data, width, color=colors_t, edgecolor='black', linewidth=0.7)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax2.set_yticks(x)
    ax2.set_yticklabels(vessels_order, fontsize=10)
    ax2.set_xlabel('Flow (mL/min)', fontsize=11, fontweight='bold')
    ax2.set_title('Abel_ref2 Reference Flows', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # The percent difference is calculated for the suptitle.
    if template_data:
        avg_pct_diff = np.mean([abs(p - t) / (abs(t) + 1) * 100 for p, t in zip(patient_data, template_data)])
        fig.suptitle(f'Patient-Specific vs Reference Model: CoW Flow Comparison (Avg Diff: {avg_pct_diff:.1f}%)',
                    fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Patient-Specific vs Reference Model: CoW Flow Comparison',
                    fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_file = output_dir / 'fig_patient_vs_template.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [COMP] fig_patient_vs_template.png")


def main():
    """
    Main entry point. Supports both interactive mode (prompts for model selection)
    and command-line mode (with --model, --results_dir, and --output_root arguments).
    """
    parser = argparse.ArgumentParser(
        description='Generate report-ready figures from FirstBlood simulation'
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Model name')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Full path to results directory')
    parser.add_argument('--output_root', type=str, default=None,
                       help='Output root directory')
    args = parser.parse_args()
    
    # Paths are built relative to the script location for portability.
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
    
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = project_root / 'pipeline' / 'output'
    
    output_dir = output_root / model_name / 'visualization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bio_analysis_dir = output_root / model_name / 'biological_analysis'
    metrics_file = bio_analysis_dir / 'global_metrics.csv'
    
    print("")
    print("=" * 80)
    print("FirstBlood Visualization Pipeline")
    print("=" * 80)
    print("Model:        {}".format(model_name))
    print("Results dir:  {}".format(results_dir))
    print("Output dir:   {}".format(output_dir))
    print("")
    
    metrics = load_global_metrics(metrics_file)
    
    if metrics is not None:
        print("Loaded metrics from biological_analysis")
        period = metrics.get('cycle_period_s', None)
    else:
        print("No biological_analysis metrics found, will estimate period")
        period = None
    
    print("\nCreating visualizations...")
    
    make_aortic_pressure_last_cycle(results_dir, output_dir, period)
    make_aortic_pressure_cycle_overlay(results_dir, output_dir, period)
    make_aortic_flow_last_cycle(results_dir, output_dir, period)
    
    if metrics is not None:
        make_summary_metrics_figure(metrics, output_dir)
    
    make_paper_style_arterial_tree(results_dir, output_dir, model_name)
    
    model_dir = project_root / 'models' / model_name
    make_multisite_waveform_panels(results_dir, model_dir, output_dir, period)
    
    # Circle of Willis visualization functions (priority 1-7) are added here.
    bio_output_dir = output_root / model_name / 'biological_analysis'
    make_cow_topology_diagram(results_dir, model_dir, bio_output_dir)
    make_cow_waveform_grid(results_dir, output_dir, period)
    
    # Patient vs template comparison is added here.
    template_results = project_root / 'projects' / 'simple_run' / 'results' / 'Abel_ref2'
    template_bio_output = output_root / 'Abel_ref2' / 'biological_analysis'
    if template_results.exists() and template_bio_output.exists():
        make_patient_template_comparison(
            bio_output_dir,
            template_bio_output,
            output_dir
        )
    
    print("")
    print("=" * 80)
    print("Visualization Complete")
    print("=" * 80)
    print("Output directory: {}".format(output_dir))
    print("")


if __name__ == "__main__":
    main()
