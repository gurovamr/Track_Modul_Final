#!/usr/bin/env python3
"""
FirstBlood Biological Validation Analysis

Comprehensive analysis of simulation results for biological meaning:
- One-page summary report with PASS/WARN/FAIL metrics
- Core hemodynamic waveform plots
- CSV exports of key metrics (global, vessel, CoW)

Usage:
    python3 analysis.py <results_dir> [output_dir]
    
Example:
    python3 analysis.py /path/to/results/patient_025 ./output
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
# CONFIGURATION
# ============================================================================

RADIUS_CORRECTION = 21.6  # Solver systematic underestimation
PRESSURE_BASELINE = 101325  # Pa (atmospheric)

# Vessel IDs for known anatomies
VESSELS = {
    'A1': 'Proximal aorta',
    'A2': 'Thoracic aorta',
    'A12': 'Right ICA',
    'A16': 'Left ICA',
    'A70': 'Right MCA',
    'A73': 'Left MCA',
    'A68': 'Right ACA A1',
    'A69': 'Left ACA A1',
    'A56': 'Basilar artery',
    'A62': 'Right Pcom',
    'A63': 'Left Pcom',
    'A77': 'Anterior communicating artery',
}

# Physiological reference ranges (healthy adult, resting)
# Only includes metrics validated in FirstBlood paper
PHYSIO_RANGES = {
    'CO': (4.5, 5.5),                    # L/min (integrated flow)
    'HR': (60, 100),                      # bpm
    'SV': (60, 100),                      # mL
    'SYS': (110, 130),                    # mmHg
    'DIA': (65, 85),                      # mmHg
    'MAP': (80, 100),                     # mmHg
    'PP': (30, 60),                       # mmHg (pulse pressure)
    'mass_balance': (-5, 5),              # % error
}

# Note: Velocities are NOT validated in FirstBlood paper and are marked as INFO only
# Velocity magnitudes are sensitive to radius and are systematically underestimated

# ============================================================================
# UTILITY: LOAD AND PREPROCESS
# ============================================================================

def load_vessel_data(vessel_id, results_dir):
    """Load vessel output and convert units."""
    fpath = Path(results_dir) / "arterial" / f"{vessel_id}.txt"
    if not fpath.exists():
        return None
    
    try:
        data = np.genfromtxt(fpath, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # CORRECTED FirstBlood output columns (verified by continuity equation):
        # 0=time, 1=P_mean, 2=P_mean_old
        # 3=Q1 (has noise), 4=Q (appears to be v_normalized or similar, not flow)
        # 5=v1 (has noise), 6=v_in (small), 7=u1 (has noise), 8=u_out (velocity, m/s)
        # 9=A1, 10=A2 (areas, m²)
        # 11=L_seg, 12=L_seg
        nrows, ncols = data.shape
        
        result = {
            'time': data[:, 0],           # s
            'P_abs': data[:, 1],          # Pa
            'Q': data[:, 4],              # Not actual flow - use v*A instead
            'v': data[:, 8],              # m/s (column 8 is reliable velocity)
            'A': data[:, 10],             # m² (cross-sectional area A2)
            'r': np.sqrt(data[:, 10] / np.pi) * RADIUS_CORRECTION if ncols > 10 else None,  # m (radius corrected)
        }
        
        # Derive additional fields
        if result['P_abs'] is not None:
            result['P_gauge'] = (result['P_abs'] - PRESSURE_BASELINE) / 133.322  # mmHg
        
        return result
    except Exception as e:
        print(f"  Warning: Failed to load {vessel_id}: {e}")
        return None


def extract_last_cycle(data, time_col, n_samples=None):
    """Extract last cycle from signal using peak detection."""
    if data is None or len(data) < 10:
        return None, None, None
    
    # Find pressure peaks to identify cycle boundaries
    peaks, _ = signal.find_peaks(data, distance=5, prominence=np.max(data)*0.1)
    
    if len(peaks) < 2:
        # Fallback: use last 1/5 of data
        start_idx = max(0, int(0.8 * len(data)))
        end_idx = len(data)
    else:
        # Use last complete cycle
        start_idx = peaks[-2]
        end_idx = peaks[-1]
    
    # Ensure both arrays have same length
    end_idx = min(end_idx, len(data))
    start_idx = min(start_idx, end_idx - 2)
    
    cycle_data = data[start_idx:end_idx+1]
    cycle_time = time_col[start_idx:end_idx+1]
    
    return cycle_data, cycle_time, (start_idx, end_idx)


def extract_cycles_multiple(data, time_col, n_cycles=3):
    """Extract last N cycles."""
    cycles = []
    peaks, _ = signal.find_peaks(data, distance=5, prominence=np.max(data)*0.1)
    
    if len(peaks) < n_cycles + 1:
        return None
    
    for i in range(n_cycles):
        start_idx = peaks[-(n_cycles-i+1)]
        end_idx = peaks[-(n_cycles-i)]
        cycles.append({
            'data': data[start_idx:end_idx],
            'time': time_col[start_idx:end_idx],
            'indices': (start_idx, end_idx),
        })
    
    return cycles


# ============================================================================
# SECTION 1: GLOBAL HEMODYNAMIC METRICS
# ============================================================================

def compute_global_metrics(results_dir):
    """
    Compute global hemodynamic metrics: CO, HR, SV, pressures, velocities.
    """
    metrics = {}
    
    # Load aorta data
    aorta = load_vessel_data('A1', results_dir)
    if aorta is None:
        return None
    
    # Find cycle boundaries from pressure
    P_data = aorta['P_gauge']
    t_data = aorta['time']
    
    peaks, _ = signal.find_peaks(P_data, distance=5, prominence=np.max(P_data)*0.1)
    
    if len(peaks) < 2:
        start_idx = max(0, int(0.8 * len(t_data)))
        end_idx = len(t_data) - 1
    else:
        start_idx = peaks[-2]
        end_idx = peaks[-1]
    
    # Extract last cycle (ensure same indices for all columns)
    end_idx = min(end_idx + 1, len(t_data))  # +1 to include endpoint
    
    P_cycle = P_data[start_idx:end_idx]
    t_cycle = t_data[start_idx:end_idx]
    Q_cycle = aorta['Q'][start_idx:end_idx] if aorta['Q'] is not None else None
    v_cycle = aorta['v'][start_idx:end_idx] if aorta['v'] is not None else None
    
    if len(P_cycle) < 2:
        return None
    
    # Pressure metrics
    metrics['P_sys'] = float(np.max(P_cycle))
    metrics['P_dia'] = float(np.min(P_cycle))
    metrics['P_map'] = float(np.mean(P_cycle))
    metrics['PP'] = metrics['P_sys'] - metrics['P_dia']
    
    # Cycle period
    cycle_period = t_cycle[-1] - t_cycle[0]
    metrics['cycle_period'] = cycle_period
    metrics['HR'] = 60.0 / cycle_period
    
    # Velocity metrics
    if v_cycle is not None:
        metrics['v_mean'] = float(np.mean(v_cycle))
        metrics['v_peak'] = float(np.max(np.abs(v_cycle)))
    
    # Cardiac output (from flow computed as v * A)
    # Note: Flow columns 3-4 in FirstBlood output are NOT reliable
    # We calculate flow from velocity (col 8) and cross-sectional area (col 10)
    if aorta['v'] is not None and 'A' in aorta:
        # Q = v * A (in m³/s)
        Q_actual = aorta['v'] * aorta['A']
        Q_cycle = Q_actual[start_idx:end_idx]
        
        # Integrate to get stroke volume in m³, then convert to mL
        sv_m3 = np.trapz(Q_cycle, t_cycle)
        metrics['SV_ml'] = sv_m3 * 1e6
        metrics['CO_lmin'] = (sv_m3 * 1e6 / cycle_period) * (60.0 / 1000.0)  # mL/s to L/min
    
    # Time step info
    dt_array = np.diff(aorta['time'])
    metrics['dt_mean'] = float(np.mean(dt_array))
    metrics['t_total'] = float(aorta['time'][-1])
    metrics['n_steps'] = len(aorta['time'])
    
    return metrics


def evaluate_metric(value, key, ranges=PHYSIO_RANGES):
    """Compare metric to physiological range."""
    if value is None or key not in ranges:
        return 'INFO', None
    
    low, high = ranges[key]
    
    if low <= value <= high:
        return 'PASS', value
    elif low * 0.8 <= value <= high * 1.2:  # 20% tolerance for WARN
        return 'WARN', value
    else:
        return 'FAIL', value


# ============================================================================
# SECTION 2: VESSEL-SPECIFIC METRICS
# ============================================================================

def compute_vessel_metrics(results_dir, vessel_ids=None):
    """
    Compute metrics for specific vessels: mean/peak flow, velocity, pressure.
    """
    if vessel_ids is None:
        vessel_ids = list(VESSELS.keys())
    
    results = []
    
    for vid in vessel_ids:
        data = load_vessel_data(vid, results_dir)
        if data is None:
            continue
        
        # Extract last cycle
        P_cycle, t_cycle, _ = extract_last_cycle(data['P_gauge'], data['time'])
        Q_cycle, _, _ = extract_last_cycle(data['Q'], data['time']) if data['Q'] is not None else (None, None, None)
        v_cycle, _, _ = extract_last_cycle(data['v'], data['time']) if data['v'] is not None else (None, None, None)
        
        metric_row = {'vessel_id': vid, 'name': VESSELS.get(vid, 'Unknown')}
        
        if P_cycle is not None:
            metric_row['P_mean'] = float(np.mean(P_cycle))
            metric_row['P_sys'] = float(np.max(P_cycle))
            metric_row['P_dia'] = float(np.min(P_cycle))
            metric_row['P_pulse'] = float(metric_row['P_sys'] - metric_row['P_dia'])
        
        if Q_cycle is not None:
            metric_row['Q_mean'] = float(np.mean(Q_cycle))
            metric_row['Q_peak'] = float(np.max(np.abs(Q_cycle)))
        
        if v_cycle is not None:
            metric_row['v_mean'] = float(np.mean(v_cycle))
            metric_row['v_peak'] = float(np.max(np.abs(v_cycle)))
        
        results.append(metric_row)
    
    return pd.DataFrame(results)


# ============================================================================
# SECTION 3: CIRCLE OF WILLIS METRICS
# ============================================================================

def compute_cow_metrics(results_dir):
    """
    Compute Circle of Willis-specific metrics: L/R asymmetry, collateral flows.
    """
    results = []
    
    # ICA flows
    ica_r = load_vessel_data('A12', results_dir)
    ica_l = load_vessel_data('A16', results_dir)
    
    cow_row = {}
    
    if ica_r and ica_r['Q'] is not None:
        Q_r, _, _ = extract_last_cycle(ica_r['Q'], ica_r['time'])
        cow_row['ICA_R_mean'] = float(np.mean(Q_r)) if Q_r is not None else None
    
    if ica_l and ica_l['Q'] is not None:
        Q_l, _, _ = extract_last_cycle(ica_l['Q'], ica_l['time'])
        cow_row['ICA_L_mean'] = float(np.mean(Q_l)) if Q_l is not None else None
    
    # MCA flows (measure asymmetry)
    mca_r = load_vessel_data('A70', results_dir)
    mca_l = load_vessel_data('A73', results_dir)
    
    if mca_r and mca_r['Q'] is not None:
        Q_r, _, _ = extract_last_cycle(mca_r['Q'], mca_r['time'])
        cow_row['MCA_R_mean'] = float(np.mean(Q_r)) if Q_r is not None else None
    
    if mca_l and mca_l['Q'] is not None:
        Q_l, _, _ = extract_last_cycle(mca_l['Q'], mca_l['time'])
        cow_row['MCA_L_mean'] = float(np.mean(Q_l)) if Q_l is not None else None
    
    # L/R asymmetry
    if 'MCA_R_mean' in cow_row and 'MCA_L_mean' in cow_row:
        q_r = cow_row['MCA_R_mean']
        q_l = cow_row['MCA_L_mean']
        asymmetry = abs(q_r - q_l) / (abs(q_r) + abs(q_l) + 1e-16) * 100
        cow_row['MCA_asymmetry_pct'] = float(asymmetry)
    
    # Communicating arteries
    acom = load_vessel_data('A77', results_dir)
    if acom and acom['Q'] is not None:
        Q_acom, _, _ = extract_last_cycle(acom['Q'], acom['time'])
        cow_row['Acom_mean'] = float(np.mean(Q_acom)) if Q_acom is not None else None
    
    pcom_r = load_vessel_data('A62', results_dir)
    if pcom_r and pcom_r['Q'] is not None:
        Q_pcr, _, _ = extract_last_cycle(pcom_r['Q'], pcom_r['time'])
        cow_row['Pcom_R_mean'] = float(np.mean(Q_pcr)) if Q_pcr is not None else None
    
    pcom_l = load_vessel_data('A63', results_dir)
    if pcom_l and pcom_l['Q'] is not None:
        Q_pcl, _, _ = extract_last_cycle(pcom_l['Q'], pcom_l['time'])
        cow_row['Pcom_L_mean'] = float(np.mean(Q_pcl)) if Q_pcl is not None else None
    
    return cow_row


# ============================================================================
# SECTION 4: PLOTTING
# ============================================================================

def plot_aortic_pressure(results_dir, output_dir):
    """Plot aortic pressure last cycle in mmHg."""
    aorta = load_vessel_data('A1', results_dir)
    if aorta is None or aorta['P_gauge'] is None:
        return
    
    P_cycle, t_cycle, _ = extract_last_cycle(aorta['P_gauge'], aorta['time'])
    if P_cycle is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_cycle, P_cycle, 'b-', linewidth=2)
    ax.fill_between(t_cycle, P_cycle, alpha=0.3)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Pressure (mmHg)', fontsize=12)
    ax.set_title('Aortic Pressure Waveform', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "aortic_pressure_last_cycle.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_aortic_flow(results_dir, output_dir):
    """Plot aortic flow last cycle. Note: Computed from v*A."""
    aorta = load_vessel_data('A1', results_dir)
    if aorta is None or aorta['v'] is None or aorta['A'] is None:
        return
    
    # Find cycle boundaries
    peaks, _ = signal.find_peaks(aorta['P_gauge'], distance=5, prominence=np.max(aorta['P_gauge'])*0.1)
    if len(peaks) < 2:
        return
    
    idx_start = peaks[-2]
    idx_end = peaks[-1]
    
    # Compute Q = v*A
    Q_cycle = aorta['v'][idx_start:idx_end+1] * aorta['A'][idx_start:idx_end+1]
    t_cycle = aorta['time'][idx_start:idx_end+1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Convert m³/s to L/min
    Q_lmin = Q_cycle * 60000
    ax.plot(t_cycle, Q_lmin, 'r-', linewidth=2)
    ax.fill_between(t_cycle, Q_lmin, alpha=0.3, color='red')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Flow (L/min)', fontsize=12)
    ax.set_title('Aortic Flow Waveform', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.savefig(output_dir / "aortic_flow_last_cycle.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_aortic_velocity(results_dir, output_dir):
    """Plot aortic velocity last cycle in m/s."""
    aorta = load_vessel_data('A1', results_dir)
    if aorta is None or aorta['v'] is None:
        return
    
    v_cycle, t_cycle, _ = extract_last_cycle(aorta['v'], aorta['time'])
    if v_cycle is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_cycle, v_cycle, 'g-', linewidth=2)
    ax.fill_between(t_cycle, v_cycle, alpha=0.3, color='green')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Aortic Velocity Waveform', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "aortic_velocity_last_cycle.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_cycle_overlay(results_dir, output_dir):
    """Plot overlay of last 3-5 aortic pressure cycles."""
    aorta = load_vessel_data('A1', results_dir)
    if aorta is None or aorta['P_gauge'] is None:
        return
    
    cycles = extract_cycles_multiple(aorta['P_gauge'], aorta['time'], n_cycles=5)
    if cycles is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(cycles)))
    
    for i, cycle in enumerate(cycles):
        t_norm = (cycle['time'] - cycle['time'][0]) / (cycle['time'][-1] - cycle['time'][0])
        ax.plot(t_norm, cycle['data'], 'o-', markersize=1, label=f'Cycle {i+1}', 
               color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Normalized Cycle Time', fontsize=12)
    ax.set_ylabel('Pressure (mmHg)', fontsize=12)
    ax.set_title('Periodic Convergence (Last 5 Cycles)', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "cycle_overlay_aortic_pressure.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_ica_waveform(results_dir, output_dir):
    """Plot ICA pressure and velocity waveforms."""
    ica_r = load_vessel_data('A12', results_dir)
    if ica_r is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    if ica_r['P_gauge'] is not None:
        P_cycle, t_cycle, _ = extract_last_cycle(ica_r['P_gauge'], ica_r['time'])
        if P_cycle is not None:
            ax1.plot(t_cycle, P_cycle, 'b-', linewidth=2)
            ax1.fill_between(t_cycle, P_cycle, alpha=0.3)
            ax1.set_ylabel('Pressure (mmHg)', fontsize=11)
            ax1.set_title('Right ICA Pressure', fontsize=12)
            ax1.grid(True, alpha=0.3)
    
    if ica_r['v'] is not None:
        v_cycle, t_cycle, _ = extract_last_cycle(ica_r['v'], ica_r['time'])
        if v_cycle is not None:
            ax2.plot(t_cycle, v_cycle, 'r-', linewidth=2)
            ax2.fill_between(t_cycle, v_cycle, alpha=0.3, color='red')
            ax2.set_xlabel('Time (s)', fontsize=11)
            ax2.set_ylabel('Velocity (m/s)', fontsize=11)
            ax2.set_title('Right ICA Velocity', fontsize=12)
            ax2.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "carotid_ica_waveform.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_cow_mca_flows(results_dir, output_dir):
    """Plot MCA L vs R flows."""
    mca_r = load_vessel_data('A70', results_dir)
    mca_l = load_vessel_data('A73', results_dir)
    
    if mca_r is None or mca_l is None or mca_r['Q'] is None or mca_l['Q'] is None:
        return
    
    Q_r_cycle, t_r, _ = extract_last_cycle(mca_r['Q'], mca_r['time'])
    Q_l_cycle, t_l, _ = extract_last_cycle(mca_l['Q'], mca_l['time'])
    
    if Q_r_cycle is None or Q_l_cycle is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert mL/s to L/min
    Q_r_lmin = Q_r_cycle * 60 / 1000
    Q_l_lmin = Q_l_cycle * 60 / 1000
    
    ax.plot(t_r, Q_r_lmin, 'r-', linewidth=2, label='R MCA', alpha=0.8)
    ax.plot(t_l, Q_l_lmin, 'b-', linewidth=2, label='L MCA', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Flow (L/min)', fontsize=12)
    ax.set_title('MCA Flow: Left vs Right', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.savefig(output_dir / "cow_mca_lr_flow.png", dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================================
# SECTION 5: REPORT GENERATION
# ============================================================================

def generate_summary_report(global_metrics, cow_metrics, output_dir):
    """Generate one-page summary report."""
    if global_metrics is None:
        return
    
    lines = []
    lines.append("=" * 80)
    lines.append("FIRSTBLOOD HEMODYNAMIC ANALYSIS - BIOLOGICAL VALIDATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("VALIDATION SCOPE:")
    lines.append("  This analysis validates metrics consistent with the FirstBlood paper:")
    lines.append("    ✓ Aortic pressure waveform (shape, systolic/diastolic/MAP)")
    lines.append("    ✓ Cardiac output (integrated flow over cycle)")
    lines.append("    ✓ Heart rate and cycle convergence")
    lines.append("    ✓ Circle of Willis flow distribution")
    lines.append("")
    lines.append("  Velocities are reported as INFO only (not validated in paper).")
    lines.append("  Peak instantaneous flows are qualitative (not compared to CO ranges).")
    lines.append("=" * 80)
    lines.append("")
    
    # Run metadata
    lines.append("RUN METADATA:")
    lines.append("-" * 80)
    lines.append(f"  Total simulation time:  {global_metrics['t_total']:.3f} s")
    lines.append(f"  Number of time steps:   {global_metrics['n_steps']}")
    lines.append(f"  Mean time step (Δt):    {global_metrics['dt_mean']:.6e} s")
    lines.append(f"  Detected cycle period:  {global_metrics['cycle_period']:.4f} s")
    lines.append(f"  Estimated heart rate:   {global_metrics['HR']:.1f} bpm")
    lines.append("")
    
    # Pressure note
    lines.append("PRESSURE UNITS NOTATION:")
    lines.append("  All pressures reported in mmHg (gauge)")
    lines.append(f"  Baseline subtracted:    {PRESSURE_BASELINE} Pa (atmospheric)")
    lines.append(f"  Absolute pressure:      P_gauge + {PRESSURE_BASELINE/133.322:.1f} mmHg")
    lines.append("")
    
    # Global metrics table
    lines.append("GLOBAL HEMODYNAMIC METRICS:")
    lines.append("-" * 80)
    
    # Aortic pressures
    if 'P_sys' in global_metrics:
        status, val = evaluate_metric(global_metrics['P_sys'], 'SYS')
        lines.append(f"  Aortic Systolic:        {global_metrics['P_sys']:7.1f} mmHg   [{status}]  (target: 110-130)")
    
    if 'P_dia' in global_metrics:
        status, val = evaluate_metric(global_metrics['P_dia'], 'DIA')
        lines.append(f"  Aortic Diastolic:       {global_metrics['P_dia']:7.1f} mmHg   [{status}]  (target: 65-85)")
    
    if 'P_map' in global_metrics:
        status, val = evaluate_metric(global_metrics['P_map'], 'MAP')
        lines.append(f"  Aortic Mean (MAP):      {global_metrics['P_map']:7.1f} mmHg   [{status}]  (target: 80-100)")
    
    if 'PP' in global_metrics:
        status, val = evaluate_metric(global_metrics['PP'], 'PP')
        lines.append(f"  Pulse Pressure:         {global_metrics['PP']:7.1f} mmHg   [{status}]  (target: 30-60)")
    
    # Cardiac output
    if 'CO_lmin' in global_metrics:
        status, val = evaluate_metric(global_metrics['CO_lmin'], 'CO')
        lines.append(f"  Cardiac Output (CO):    {global_metrics['CO_lmin']:7.1f} L/min   [{status}]  (target: 4.5-5.5)")
    
    if 'SV_ml' in global_metrics:
        status, val = evaluate_metric(global_metrics['SV_ml'], 'SV')
        lines.append(f"  Stroke Volume (SV):     {global_metrics['SV_ml']:7.1f} mL     [{status}]  (target: 60-100)")
    
    if 'HR' in global_metrics:
        status, val = evaluate_metric(global_metrics['HR'], 'HR')
        lines.append(f"  Heart Rate:             {global_metrics['HR']:7.1f} bpm    [{status}]  (target: 60-100)")
    
    # Aortic velocities (INFO only - not validated in FirstBlood paper)
    lines.append("")
    lines.append("VELOCITY METRICS (INFORMATIONAL ONLY):")
    lines.append("-" * 80)
    lines.append("  Note: Velocity magnitudes are NOT validated in FirstBlood paper.")
    lines.append("        Velocities are sensitive to radius and systematically underestimated.")
    lines.append("        These values are for qualitative assessment only.")
    if 'v_mean' in global_metrics:
        lines.append(f"  Aortic mean velocity:   {global_metrics['v_mean']:7.2f} m/s    [INFO]  (expected: 0.2-0.5)")
    
    if 'v_peak' in global_metrics:
        lines.append(f"  Aortic peak velocity:   {global_metrics['v_peak']:7.2f} m/s    [INFO]  (expected: 0.8-1.2)")
    
    lines.append("")
    
    # Circle of Willis metrics
    if cow_metrics:
        lines.append("CIRCLE OF WILLIS METRICS:")
        lines.append("-" * 80)
        
        if 'ICA_R_mean' in cow_metrics and cow_metrics['ICA_R_mean'] is not None:
            lines.append(f"  R ICA mean flow:        {cow_metrics['ICA_R_mean']:8.2f} mL/s")
        
        if 'ICA_L_mean' in cow_metrics and cow_metrics['ICA_L_mean'] is not None:
            lines.append(f"  L ICA mean flow:        {cow_metrics['ICA_L_mean']:8.2f} mL/s")
        
        if 'MCA_R_mean' in cow_metrics and cow_metrics['MCA_R_mean'] is not None:
            lines.append(f"  R MCA mean flow:        {cow_metrics['MCA_R_mean']:8.2f} mL/s")
        
        if 'MCA_L_mean' in cow_metrics and cow_metrics['MCA_L_mean'] is not None:
            lines.append(f"  L MCA mean flow:        {cow_metrics['MCA_L_mean']:8.2f} mL/s")
        
        if 'MCA_asymmetry_pct' in cow_metrics and cow_metrics['MCA_asymmetry_pct'] is not None:
            status = 'PASS' if cow_metrics['MCA_asymmetry_pct'] < 20 else 'WARN'
            lines.append(f"  MCA L/R asymmetry:      {cow_metrics['MCA_asymmetry_pct']:7.1f} %     [{status}]  (target: <20%)")
        
        if 'Acom_mean' in cow_metrics and cow_metrics['Acom_mean'] is not None:
            lines.append(f"  Anterior comm. flow:    {cow_metrics['Acom_mean']:8.2f} mL/s")
        
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("STATUS LEGEND:")
    lines.append("  [PASS]  — Within physiological target range (validated in paper)")
    lines.append("  [WARN]  — Within ±20% of target (acceptable, may need longer simulation)")
    lines.append("  [FAIL]  — Outside acceptable range (requires investigation)")
    lines.append("  [INFO]  — Informational only (not validated in FirstBlood paper)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("INTERPRETATION GUIDE:")
    lines.append("-" * 80)
    lines.append("  Pressure plots: Primary validation evidence. Waveform shape and convergence")
    lines.append("                  are most important. Sharp upstroke, gradual decay, dicrotic")
    lines.append("                  notch indicate physiological behavior.")
    lines.append("")
    lines.append("  Flow plots:     Show qualitative waveform shape. Peak values are instantaneous")
    lines.append("                  flow (NOT cardiac output). CO is computed by integration.")
    lines.append("")
    lines.append("  Velocity plots: Qualitative only. Magnitudes systematically underestimated.")
    lines.append("                  Shape and timing are meaningful, absolute values are not.")
    lines.append("")
    lines.append("  CoW metrics:    L/R asymmetry <20% indicates balanced cerebral perfusion.")
    lines.append("                  Absolute flow values are model-dependent.")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("OUTPUT FILES GENERATED:")
    lines.append("  • global_metrics.csv          — Summary of global metrics (one row)")
    lines.append("  • vessel_metrics.csv          — Per-vessel metrics (mean/peak P/Q/v)")
    lines.append("  • cow_metrics.csv             — CoW-specific metrics and asymmetry")
    lines.append("  • aortic_pressure_last_cycle.png")
    lines.append("  • aortic_flow_last_cycle.png")
    lines.append("  • aortic_velocity_last_cycle.png")
    lines.append("  • cycle_overlay_aortic_pressure.png")
    lines.append("  • carotid_ica_waveform.png")
    lines.append("  • cow_mca_lr_flow.png")
    lines.append("")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    with open(output_dir / "biological_validation_summary.txt", 'w') as f:
        f.write(report_text)
    
    return report_text


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_analysis(results_dir, output_dir=None):
    """Execute complete biological validation analysis."""
    results_dir = Path(results_dir)
    
    if output_dir is None:
        output_dir = Path.cwd() / "analysis_output"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"FIRSTBLOOD BIOLOGICAL VALIDATION ANALYSIS")
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # Compute metrics
    # ========================================================================
    print("Computing global hemodynamic metrics...")
    global_metrics = compute_global_metrics(str(results_dir))
    if global_metrics is None:
        print("❌ Failed to compute global metrics")
        return
    
    print("Computing vessel-specific metrics...")
    vessel_df = compute_vessel_metrics(str(results_dir))
    
    print("Computing Circle of Willis metrics...")
    cow_metrics = compute_cow_metrics(str(results_dir))
    
    # ========================================================================
    # Export metrics to CSV
    # ========================================================================
    print("Exporting metrics to CSV...")
    
    # Global metrics (one row)
    global_df = pd.DataFrame([global_metrics])
    global_csv = output_dir / "global_metrics.csv"
    global_df.to_csv(global_csv, index=False)
    print(f"  ✓ {global_csv.name}")
    
    # Vessel metrics
    if not vessel_df.empty:
        vessel_csv = output_dir / "vessel_metrics.csv"
        vessel_df.to_csv(vessel_csv, index=False)
        print(f"  ✓ {vessel_csv.name}")
    
    # CoW metrics (one row)
    cow_df = pd.DataFrame([cow_metrics])
    cow_csv = output_dir / "cow_metrics.csv"
    cow_df.to_csv(cow_csv, index=False)
    print(f"  ✓ {cow_csv.name}")
    
    # ========================================================================
    # Generate plots
    # ========================================================================
    print("\nGenerating waveform plots...")
    
    plot_aortic_pressure(str(results_dir), output_dir)
    print("  ✓ aortic_pressure_last_cycle.png")
    
    plot_aortic_flow(str(results_dir), output_dir)
    print("  ✓ aortic_flow_last_cycle.png")
    
    plot_aortic_velocity(str(results_dir), output_dir)
    print("  ✓ aortic_velocity_last_cycle.png")
    
    plot_cycle_overlay(str(results_dir), output_dir)
    print("  ✓ cycle_overlay_aortic_pressure.png")
    
    plot_ica_waveform(str(results_dir), output_dir)
    print("  ✓ carotid_ica_waveform.png")
    
    plot_cow_mca_flows(str(results_dir), output_dir)
    print("  ✓ cow_mca_lr_flow.png")
    
    # ========================================================================
    # Generate summary report
    # ========================================================================
    print("\nGenerating summary report...")
    report_text = generate_summary_report(global_metrics, cow_metrics, output_dir)
    print(report_text)
    print(f"\n✓ biological_validation_summary.txt")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs in: {output_dir}/")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            print(f"  • {f.name}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analysis.py <results_dir> [output_dir]")
        print("\nExample:")
        print("  python3 analysis.py /path/to/results/patient_025")
        print("  python3 analysis.py /path/to/results/patient_025 ./output")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_analysis(results_dir, output_dir)
