#!/usr/bin/env python3

import numpy as np
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
    List available model directories in results folder.
    Returns sorted list of model names.
    """
    if not results_base_path.exists():
        return []
    
    models = [d.name for d in results_base_path.iterdir() 
              if d.is_dir()]
    return sorted(models)


def select_model_interactive(results_base_path: Path) -> Optional[str]:
    """
    List available models and prompt user to select.
    Returns selected model name or None if cancelled.
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
    Load timeseries from text file.
    Tries comma delimiter first, then whitespace.
    Returns None on failure.
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
    Detect which column contains strictly increasing time values.
    Returns column index or None if not found.
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
    Select the most pulsatile column (largest peak-to-peak amplitude).
    Excludes time column and nearly constant signals.
    Returns column index or None if not found.
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
    Resample signal to uniform time grid using linear interpolation.
    """
    t_uniform = np.linspace(time_col[0], time_col[-1], n_points)
    s_uniform = np.interp(t_uniform, time_col, signal_col)
    
    return t_uniform, s_uniform


def numpy_find_peaks(signal: np.ndarray, distance: int = 5, height: float = 0.0) -> np.ndarray:
    """
    Find peaks in signal using numpy only.
    Returns array of peak indices.
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
    Estimate cardiac period using autocorrelation.
    Searches within [min_period, max_period] range.
    Returns period in seconds or None if not found.
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
    Extract last complete cycle from signal using known period.
    Returns (cycle_signal, cycle_time) or (None, None).
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
    Extract last n complete cycles from signal.
    Returns list of (cycle_signal, cycle_time) tuples.
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
    Extract last cycle from signal using time-window slicing.
    Returns last 'period' seconds of data.
    Returns (cycle_signal, cycle_time_relative) or (None, None).
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
    Load metrics from biological_analysis global_metrics.csv.
    Returns dict with metric names and values, or None if not available.
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
    Generate figure 1: Aortic pressure last cycle.
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
    Generate figure 2: Overlay of last 3-5 cycles.
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
    Generate figure 3: Aortic flow last cycle.
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
    Generate figure 4: Summary metrics table.
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
    Read a single column from a CSV file.
    Returns list of values or None if not found.
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
    Read entire CSV file as list of dictionaries.
    Returns list of rows or None on error.
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
    Search for an arterial signal file matching a given name.
    Looks in results_dir and results_dir/arterial/ for A*.txt or name*.txt files.
    Returns full path or None if not found.
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
    Compute distance from root node to all nodes in a graph (simple BFS).
    edges_list: list of (node1, node2) tuples
    root_node: starting node for BFS
    Returns dict mapping node -> distance
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
    Generate figure 5: Paper-style arterial tree with anatomical layout.
    Head at top, torso in middle, legs at bottom.
    Vessels colored by mean flow direction (red forward, blue reversed, gray minimal).
    """
    M3S_TO_MLMIN = 60.0 * 1e6
    eps = 0.01
    
    def load_mean_flow_robust(vessel_id):
        """Load mean flow with robust unit handling"""
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
        """Blue if reversed, red forward, gray minimal"""
        if flow < -eps:
            return 'blue', 2.2
        elif abs(flow) <= eps:
            return '#bbbbbb', 0.9
        else:
            return 'red', 1.6
    
    def draw_vessel(ax, pts, vid):
        """Draw vessel as polyline, color by flow"""
        flow = load_mean_flow_robust(vid)
        color, lw = vessel_color(flow)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle='round',
                zorder=2, alpha=0.85)
        return flow
    
    def junc(ax, x, y):
        """White open circle junction node"""
        ax.plot(x, y, 'o', color='white', markersize=6,
                markeredgecolor='black', markeredgewidth=0.9, zorder=4)
    
    def term(ax, x, y):
        """Gray filled terminal node"""
        ax.plot(x, y, 'o', color='#777777', markersize=4.5, zorder=4)
    
    fig, ax = plt.subplots(figsize=(9, 17))
    ax.set_xlim(-9, 9)
    ax.set_ylim(-1, 39)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.suptitle(f'Arterial Network: {model_name}',
                 fontsize=14, fontweight='bold', y=0.99)
    
    # HEAD (y = 26 .. 38) - TOP of figure
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
    
    draw_vessel(ax, [r_cca_top, r_ica], 'A12')
    draw_vessel(ax, [r_cca_top, r_eca], 'A13')
    junc(ax, *r_cca_top)
    junc(ax, *r_ica)
    
    draw_vessel(ax, [l_cca_top, l_ica], 'A16')
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
    
    draw_vessel(ax, [ba_mid, ba_bif], 'A59')
    junc(ax, *ba_bif)
    
    r_pca_flow = draw_vessel(ax, [ba_bif, r_p1], 'A60')
    junc(ax, *r_p1)
    draw_vessel(ax, [ba_bif, l_p1], 'A61')
    junc(ax, *l_p1)
    
    draw_vessel(ax, [r_p1, r_pcoa_jxn], 'A62')
    draw_vessel(ax, [l_p1, l_pcoa_jxn], 'A63')
    
    draw_vessel(ax, [r_p1, r_pca_p2], 'A64')
    term(ax, *r_pca_p2)
    draw_vessel(ax, [l_p1, l_pca_p2], 'A65')
    term(ax, *l_pca_p2)
    
    draw_vessel(ax, [r_m1, r_mca_end], 'A70')
    junc(ax, *r_mca_end)
    draw_vessel(ax, [r_mca_end, r_mca_m2a], 'A71')
    term(ax, *r_mca_m2a)
    draw_vessel(ax, [r_mca_end, r_mca_m2b], 'A72')
    term(ax, *r_mca_m2b)
    
    draw_vessel(ax, [l_m1, l_mca_end], 'A73')
    junc(ax, *l_mca_end)
    draw_vessel(ax, [l_mca_end, l_mca_m2a], 'A74')
    term(ax, *l_mca_m2a)
    draw_vessel(ax, [l_mca_end, l_mca_m2b], 'A75')
    term(ax, *l_mca_m2b)
    
    draw_vessel(ax, [r_m1, r_aca_a1], 'A68')
    junc(ax, *r_aca_a1)
    draw_vessel(ax, [r_aca_a1, r_aca_a2], 'A76')
    term(ax, *r_aca_a2)
    
    draw_vessel(ax, [l_m1, l_aca_a1], 'A69')
    junc(ax, *l_aca_a1)
    draw_vessel(ax, [l_aca_a1, l_aca_a2], 'A78')
    term(ax, *l_aca_a2)
    
    draw_vessel(ax, [l_aca_a1, r_aca_a1], 'A77')
    
    draw_vessel(ax, [(0.75, 30.0), (1.3, 30.8)], 'A100')
    term(ax, 1.3, 30.8)
    draw_vessel(ax, [(-0.75, 30.0), (-1.3, 30.8)], 'A102')
    term(ax, -1.3, 30.8)
    
    if r_pca_flow < 0:
        ax.annotate('FLOW REVERSAL\n(Fetal variant)',
                    xy=(0.5, 29.8), xytext=(3.0, 29.0),
                    fontsize=7, fontweight='bold', color='blue',
                    bbox=dict(boxstyle='round', fc='cyan', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    zorder=6)
    
    # TORSO (y = 13 .. 26)
    heart = (0, 24.5)
    ax.plot(*heart, 'o', color='red', markersize=9, zorder=5)
    
    n1 = (0, 25.2)
    n2 = (0, 25.8)
    draw_vessel(ax, [heart, n1], 'A1')
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
    draw_vessel(ax, [n6, r_cca_bot], 'A5')
    junc(ax, *r_cca_bot)
    ax.plot([r_cca_bot[0], r_cca_top[0]], [r_cca_bot[1], r_cca_top[1]],
            color='red', linewidth=1.6, alpha=0.85, zorder=2)
    
    l_cca_bot = (-1.2, 26.2)
    draw_vessel(ax, [n3, l_cca_bot], 'A15')
    junc(ax, *l_cca_bot)
    ax.plot([l_cca_bot[0], l_cca_top[0]], [l_cca_bot[1], l_cca_top[1]],
            color='red', linewidth=1.6, alpha=0.85, zorder=2)
    
    n7 = (2.5, 24.8)
    draw_vessel(ax, [n6, n7], 'A4')
    junc(ax, *n7)
    draw_vessel(ax, [n7, (1.5, 26.0), vert_conf], 'A6')
    
    n10 = (-2.8, 24.8)
    draw_vessel(ax, [n4, n10], 'A19')
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
    
    # LEGS (y = 0 .. 14)
    n17 = (1.5, 13.0)
    draw_vessel(ax, [n13, n17], 'A42')
    junc(ax, *n17)
    draw_vessel(ax, [n17, (0.8, 12.5)], 'A45')
    term(ax, 0.8, 12.5)
    
    n18 = (2.0, 11.5)
    draw_vessel(ax, [n17, n18], 'A44')
    junc(ax, *n18)
    draw_vessel(ax, [n18, (2.8, 11.2)], 'A47')
    term(ax, 2.8, 11.2)
    
    n19 = (2.2, 9.5)
    draw_vessel(ax, [n18, n19], 'A46')
    junc(ax, *n19)
    
    draw_vessel(ax, [n19, (2.1, 6.0)], 'A48')
    term(ax, 2.1, 6.0)
    draw_vessel(ax, [n19, (2.8, 6.0)], 'A49')
    term(ax, 2.8, 6.0)
    
    n14 = (-1.5, 13.0)
    draw_vessel(ax, [n13, n14], 'A43')
    junc(ax, *n14)
    draw_vessel(ax, [n14, (-0.8, 12.5)], 'A51')
    term(ax, -0.8, 12.5)
    
    n15 = (-2.0, 11.5)
    draw_vessel(ax, [n14, n15], 'A50')
    junc(ax, *n15)
    draw_vessel(ax, [n15, (-2.8, 11.2)], 'A53')
    term(ax, -2.8, 11.2)
    
    n16 = (-2.2, 9.5)
    draw_vessel(ax, [n15, n16], 'A52')
    junc(ax, *n16)
    
    draw_vessel(ax, [n16, (-2.1, 6.0)], 'A54')
    term(ax, -2.1, 6.0)
    draw_vessel(ax, [n16, (-2.8, 6.0)], 'A55')
    term(ax, -2.8, 6.0)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Forward flow'),
        Line2D([0], [0], color='blue', linewidth=2, label='Reversed flow'),
        Line2D([0], [0], color='#bbbbbb', linewidth=1, label='Minimal flow'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=7, linewidth=0, label='Junction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#777777',
               markersize=5, linewidth=0, label='Terminal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, linewidth=0, label='Heart'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5,
              framealpha=0.95)
    
    plt.tight_layout()
    output_file = output_dir / 'fig5_arterial_tree_paper_style.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("  [5] fig5_arterial_tree_paper_style.png")


def make_network_schematic(model_dir: Path, output_dir: Path):
    """
    Generate auto-layout network topology schematic (optional).
    Reads arterial.csv and draws a simplified network diagram.
    """
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


def make_subsystem_routing_diagram(output_dir: Path):
    """
    Generate figure 6: Subsystem routing diagram.
    Shows conceptual flow: Heart -> Body -> {Brain, Periphery}
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    box_width = 1.5
    box_height = 1.2
    
    heart_x, heart_y = 1.5, 7
    body_x, body_y = 5, 7
    brain_x, brain_y = 5, 4
    periph_x, periph_y = 8, 4
    
    def draw_box(ax, x, y, label, color='lightblue'):
        rect = plt.Rectangle((x - box_width/2, y - box_height/2),
                             box_width, box_height,
                             fill=True, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=13, fontweight='bold', ha='center', va='center')
    
    def draw_arrow(ax, x1, y1, x2, y2):
        ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.3, head_length=0.2,
                fc='black', ec='black', linewidth=2)
    
    draw_box(ax, heart_x, heart_y, 'Heart', 'lightyellow')
    draw_box(ax, body_x, body_y, 'Body', 'lightgreen')
    draw_box(ax, brain_x, brain_y, 'Brain', 'lightcyan')
    draw_box(ax, periph_x, periph_y, 'Periphery', 'lightcoral')
    
    draw_arrow(ax, heart_x + box_width/2, heart_y, body_x - box_width/2, body_y)
    draw_arrow(ax, body_x, body_y - box_height/2 - 0.3, brain_x, brain_y + box_height/2 + 0.3)
    draw_arrow(ax, body_x + 0.5, body_y - box_height/2 - 0.3, periph_x - 0.5, periph_y + box_height/2 + 0.3)
    
    ax.text(5, 9.5, 'FirstBlood Systemic Circulation', fontsize=16, fontweight='bold', ha='center')
    
    fig.tight_layout()
    output_file = output_dir / 'fig6_subsystem_routing.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [6] fig6_subsystem_routing.png")


def make_multisite_waveform_panels(results_dir: Path, model_dir: Path, output_dir: Path, period: Optional[float] = None):
    """
    Generate figure 7: Multi-site waveform panels.
    Shows pressure and flow/velocity at 5 arterial locations.
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
    
    print("  [7] fig7_multisite_waveforms.png")


def main():
    """
    Main entry point with interactive and CLI modes.
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
    print("FIRSTBLOOD VISUALIZATION")
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
    
    print("\nGenerating figures...")
    
    make_aortic_pressure_last_cycle(results_dir, output_dir, period)
    make_aortic_pressure_cycle_overlay(results_dir, output_dir, period)
    make_aortic_flow_last_cycle(results_dir, output_dir, period)
    
    if metrics is not None:
        make_summary_metrics_figure(metrics, output_dir)
    
    make_paper_style_arterial_tree(results_dir, output_dir, model_name)
    
    make_subsystem_routing_diagram(output_dir)
    
    model_dir = project_root / 'models' / model_name
    make_multisite_waveform_panels(results_dir, model_dir, output_dir, period)
    
    print("")
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print("Output directory: {}".format(output_dir))
    print("")


if __name__ == "__main__":
    main()
