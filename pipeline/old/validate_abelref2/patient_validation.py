#!/usr/bin/env python3
"""
FirstBlood Patient-Specific Validation & Analysis Pipeline
Comprehensive schema discovery, numerical validation, physiological checks, and CoW analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("/home/maryyds/final/first_blood/projects/simple_run/results/patient_025")
MODEL_DIR = Path("/home/maryyds/final/first_blood/models/Abel_ref2")
OUTPUT_DIR = Path("/home/maryyds/final/first_blood/validate_abelref2/validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# CoW vessel mapping (FirstBlood IDs to anatomical names)
COW_MAPPING = {
    'A12': ('R', 'ICA'),    'A16': ('L', 'ICA'),
    'A68': ('R', 'A1'),     'A69': ('L', 'A1'),
    'A70': ('R', 'MCA'),    'A73': ('L', 'MCA'),
    'A76': ('R', 'A2'),     'A78': ('L', 'A2'),
    'A60': ('R', 'P1'),     'A61': ('L', 'P1'),
    'A62': ('R', 'Pcom'),   'A63': ('L', 'Pcom'),
    'A64': ('R', 'P2'),     'A65': ('L', 'P2'),
    'A56': ('BA', 'BA'),    'A59': ('BA', 'BA'),
    'A77': (None, 'Acom'),
}

# Expected vessel radii ranges (meters) for identification
VESSEL_RANGES = {
    'aorta': (0.010, 0.020),
    'carotid': (0.004, 0.008),
    'cerebral': (0.0005, 0.003),
}

# ============================================================================
# 1. SCHEMA DISCOVERY
# ============================================================================

def discover_arterial_schema() -> Dict[str, Dict[str, Any]]:
    """
    Discover schema for all arterial output files.
    Infer columns by analyzing data magnitude and consistency.
    """
    schema = {}
    arterial_dir = RESULTS_DIR / "arterial"
    
    if not arterial_dir.exists():
        print(f"❌ Arterial directory not found: {arterial_dir}")
        return schema
    
    for txt_file in sorted(arterial_dir.glob("*.txt"))[:103]:  # All 103 vessels
        try:
            data = np.loadtxt(txt_file, delimiter=',')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            vessel_id = txt_file.stem
            n_cols = data.shape[1]
            
            # Infer columns: time, p_start, p_end, q_start, q_end, u_start, u_end, a_start, a_end, r_start, r_end, ...
            # Expected: time (monotonic), pressures (~100000), flows (~0-0.01), velocities (~0-2), areas/radius
            
            col_info = {
                'file': str(txt_file),
                'shape': data.shape,
                'n_cols': n_cols,
                'columns': {}
            }
            
            # Col 0: time (always monotonic)
            col_info['columns'][0] = {
                'name': 'time',
                'unit': 's',
                'range': (data[0, 0], data[-1, 0]),
                'mean': float(np.mean(data[:, 0]))
            }
            
            # Identify remaining columns by magnitude
            for col in range(1, n_cols):
                col_data = data[:, col]
                col_mean = float(np.mean(col_data))
                col_std = float(np.std(col_data))
                col_max = float(np.max(col_data))
                col_min = float(np.min(col_data))
                
                # Classify by magnitude
                if 95000 < col_mean < 110000:  # Pressure (absolute)
                    col_type = 'pressure_abs'
                elif col_mean < 100:
                    if col_max < 0.1:
                        col_type = 'flow_or_area'  # Could be flow (m^3/s) or area
                    else:
                        col_type = 'velocity_or_residual'
                else:
                    col_type = 'unknown'
                
                col_info['columns'][col] = {
                    'name': col_type,
                    'mean': col_mean,
                    'std': col_std,
                    'min': col_min,
                    'max': col_max,
                    'range': col_max - col_min
                }
            
            schema[vessel_id] = col_info
            
        except Exception as e:
            print(f"⚠️  Failed to read {txt_file.stem}: {e}")
    
    return schema

def identify_aorta(schema: Dict[str, Dict]) -> Optional[str]:
    """Identify aortic vessel by high radius and large diameter."""
    candidates = []
    for vessel_id, info in schema.items():
        if info['n_cols'] >= 11:  # Full MOC output
            r_start_idx = 9  # Typical location
            if r_start_idx < info['n_cols']:
                r_mean = info['columns'].get(r_start_idx, {}).get('mean', 0)
                if VESSEL_RANGES['aorta'][0] <= r_mean <= VESSEL_RANGES['aorta'][1]:
                    candidates.append((vessel_id, r_mean))
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return 'A1'  # Fallback

def load_arterial_data(vessel_id: str) -> Optional[np.ndarray]:
    """Load data from a single arterial vessel file."""
    path = RESULTS_DIR / "arterial" / f"{vessel_id}.txt"
    if path.exists():
        try:
            return np.loadtxt(path, delimiter=',')
        except:
            return None
    return None

# ============================================================================
# 2. PRESSURE CONVERSION & COLUMN EXTRACTION
# ============================================================================

def extract_vessel_signals(vessel_data: np.ndarray, n_cols_expected: int = 13) -> Dict[str, np.ndarray]:
    """
    Extract time, pressure, flow, velocity from vessel MOC output.
    CORRECTED column layout for FirstBlood output (from data inspection):
    Col 0: time (s)
    Col 1: p_start (Pa, absolute)
    Col 2: p_end (Pa, absolute)
    Col 3: q_start (m^3/s)
    Col 4: q_end (m^3/s)
    Col 5: u_start (m/s)
    Col 6: u_end (m/s)
    Col 7: a_start (m^2)
    Col 8: a_end (m^2)
    Col 9: r_start (m)
    Col 10: r_end (m)
    Col 11-12: diagnostic parameters
    """
    if vessel_data.ndim == 1:
        vessel_data = vessel_data.reshape(1, -1)
    
    signals = {}
    signals['time'] = vessel_data[:, 0]
    
    # Pressure columns (absolute ~100000 Pa)
    if vessel_data.shape[1] >= 3:
        signals['p_start_abs'] = vessel_data[:, 1]
        signals['p_end_abs'] = vessel_data[:, 2]
        # Convert to gauge (mmHg)
        signals['p_start_gauge'] = (signals['p_start_abs'] - 100000) / 133.322  # Pa to mmHg
        signals['p_end_gauge'] = (signals['p_end_abs'] - 100000) / 133.322
    
    # Flow columns (m^3/s)
    if vessel_data.shape[1] >= 5:
        signals['q_start'] = vessel_data[:, 3]  # m^3/s
        signals['q_end'] = vessel_data[:, 4]
    
    # Velocity columns (m/s)
    if vessel_data.shape[1] >= 7:
        signals['u_start'] = vessel_data[:, 5]  # m/s
        signals['u_end'] = vessel_data[:, 6]
    
    # Area columns (m^2)
    if vessel_data.shape[1] >= 9:
        signals['a_start'] = vessel_data[:, 7]  # m^2
        signals['a_end'] = vessel_data[:, 8]
    
    # Radius columns (m)
    if vessel_data.shape[1] >= 11:
        signals['r_start'] = vessel_data[:, 9]  # m
        signals['r_end'] = vessel_data[:, 10]
    
    return signals

# ============================================================================
# 3. NUMERICAL CORRECTNESS
# ============================================================================

def check_nan_inf(signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Check for NaN, Inf, and extreme values."""
    issues = {'nan_count': 0, 'inf_count': 0, 'extreme_values': [], 'ok': True}
    
    for key, data in signals.items():
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        if nan_count > 0:
            issues['nan_count'] += nan_count
            issues['ok'] = False
        if inf_count > 0:
            issues['inf_count'] += inf_count
            issues['ok'] = False
        
        # Check for extreme blow-ups (>10x expected range)
        if 'pressure' in key:
            if np.max(np.abs(data)) > 5e5:  # Pressure should not exceed ~500 kPa
                issues['extreme_values'].append(f"{key}: max={np.max(data):.2e}")
                issues['ok'] = False
        elif 'flow' in key:
            if np.max(np.abs(data)) > 1.0:  # Flow shouldn't exceed 1 m^3/s in most vessels
                issues['extreme_values'].append(f"{key}: max={np.max(data):.2e}")
    
    return issues

def compute_convergence(signals: Dict[str, np.ndarray], cycle_time: float = 1.0) -> Dict[str, Any]:
    """
    Compute cycle-to-cycle convergence for the last 5 cycles.
    Uses aortic pressure (p_start_gauge) as reference.
    """
    convergence = {'status': 'UNKNOWN', 'rms_error_pct': None, 'cycles_analyzed': 0, 'notes': []}
    
    if 'p_start_gauge' not in signals:
        convergence['notes'].append("No gauge pressure signal available")
        return convergence
    
    time = signals['time']
    pressure = signals['p_start_gauge']
    
    # Estimate cycle period from data length
    if len(time) < 100:
        convergence['notes'].append("Insufficient time steps for convergence analysis")
        return convergence
    
    # Assume last ~5 cycles are available
    dt = time[1] - time[0]
    samples_per_cycle = int(cycle_time / dt)
    
    if samples_per_cycle < 10:
        convergence['notes'].append(f"Cycle too short ({samples_per_cycle} samples); insufficient resolution")
        return convergence
    
    # Extract last 5 cycles
    n_cycles = min(5, len(pressure) // samples_per_cycle)
    if n_cycles < 2:
        convergence['notes'].append(f"Fewer than 2 cycles available")
        return convergence
    
    cycles = []
    for i in range(n_cycles):
        start_idx = len(pressure) - (n_cycles - i) * samples_per_cycle
        end_idx = start_idx + samples_per_cycle
        if end_idx <= len(pressure):
            cycles.append(pressure[start_idx:end_idx])
    
    # Compute RMS error between last cycle and second-to-last
    if len(cycles) >= 2:
        ref_cycle = cycles[-2]
        last_cycle = cycles[-1]
        
        # Normalize to same length
        min_len = min(len(ref_cycle), len(last_cycle))
        ref_cycle_norm = (ref_cycle[:min_len] - np.min(ref_cycle[:min_len])) / (np.max(ref_cycle[:min_len]) - np.min(ref_cycle[:min_len]) + 1e-10)
        last_cycle_norm = (last_cycle[:min_len] - np.min(last_cycle[:min_len])) / (np.max(last_cycle[:min_len]) - np.min(last_cycle[:min_len]) + 1e-10)
        
        rms_error = np.sqrt(np.mean((ref_cycle_norm - last_cycle_norm) ** 2))
        rms_error_pct = rms_error * 100
        
        convergence['rms_error_pct'] = rms_error_pct
        convergence['cycles_analyzed'] = n_cycles
        convergence['status'] = 'PASS' if rms_error_pct < 0.1 else ('WARN' if rms_error_pct < 1.0 else 'FAIL')
        convergence['notes'].append(f"RMS error: {rms_error_pct:.3f}%")
    
    return convergence

def check_mass_conservation(aorta_signals: Dict, heart_signals: Dict) -> Dict[str, Any]:
    """
    Check mass conservation: CO_in (from heart) ≈ CO_out (aorta start flow).
    """
    conservation = {'status': 'UNKNOWN', 'co_in_lmin': None, 'co_out_lmin': None, 'error_pct': None, 'notes': []}
    
    # Get aortic outflow (last cycle mean)
    if 'q_start' not in aorta_signals:
        conservation['notes'].append("No flow signal in aorta")
        return conservation
    
    q_aorta = aorta_signals['q_start']
    if len(q_aorta) < 100:
        conservation['notes'].append("Insufficient time steps")
        return conservation
    
    # Use last cycle (assume cyclic after sufficient time)
    samples_per_cycle = max(100, len(q_aorta) // 10)
    last_cycle = q_aorta[-samples_per_cycle:]
    co_out = np.mean(last_cycle) * 60000  # m^3/s to L/min
    
    # Get heart output (from heart model if available)
    if 'q_lv' in heart_signals:
        q_lv = heart_signals['q_lv']
        last_cycle_lv = q_lv[-samples_per_cycle:] if len(q_lv) > samples_per_cycle else q_lv
        co_in = np.mean(last_cycle_lv) * 60000
    else:
        conservation['notes'].append("Heart flow not available; skipping conservation check")
        return conservation
    
    conservation['co_in_lmin'] = co_in
    conservation['co_out_lmin'] = co_out
    
    if co_in > 0:
        error_pct = abs(co_out - co_in) / co_in * 100
        conservation['error_pct'] = error_pct
        conservation['status'] = 'PASS' if error_pct < 5 else ('WARN' if error_pct < 10 else 'FAIL')
        conservation['notes'].append(f"Error: {error_pct:.2f}%")
    
    return conservation

# ============================================================================
# 4. GLOBAL PHYSIOLOGY
# ============================================================================

def compute_cardiac_output(signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute cardiac output from aortic flow (last cycle mean)."""
    co = {'co_lmin': None, 'status': 'UNKNOWN', 'notes': []}
    
    if 'q_start' not in signals:
        co['notes'].append("No flow signal available")
        return co
    
    q = signals['q_start']
    if len(q) < 100:
        co['notes'].append("Insufficient data")
        return co
    
    # Use last cycle
    samples_per_cycle = max(100, len(q) // 10)
    last_cycle = q[-samples_per_cycle:]
    mean_flow = np.mean(last_cycle)
    co_lmin = mean_flow * 60000  # m^3/s to L/min
    
    co['co_lmin'] = co_lmin
    co['status'] = 'PASS' if 4 <= co_lmin <= 6 else ('WARN' if 3 <= co_lmin <= 7 else 'FAIL')
    co['notes'].append(f"CO = {co_lmin:.2f} L/min (normal 4-6)")
    
    return co

def compute_aortic_pressure_stats(signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute aortic pressure statistics (systolic, diastolic, mean)."""
    stats = {'systolic_mmhg': None, 'diastolic_mmhg': None, 'mean_mmhg': None, 'status': 'UNKNOWN', 'notes': []}
    
    if 'p_start_gauge' not in signals:
        stats['notes'].append("No gauge pressure available")
        return stats
    
    p = signals['p_start_gauge']
    if len(p) < 100:
        stats['notes'].append("Insufficient data")
        return stats
    
    # Use last cycle
    samples_per_cycle = max(100, len(p) // 10)
    last_cycle = p[-samples_per_cycle:]
    
    sys_mmhg = np.max(last_cycle)
    dia_mmhg = np.min(last_cycle)
    mean_mmhg = np.mean(last_cycle)
    
    stats['systolic_mmhg'] = sys_mmhg
    stats['diastolic_mmhg'] = dia_mmhg
    stats['mean_mmhg'] = mean_mmhg
    
    # Check ranges
    sys_ok = 110 <= sys_mmhg <= 130
    dia_ok = 65 <= dia_mmhg <= 85
    mean_ok = 90 <= mean_mmhg <= 100
    
    if sys_ok and dia_ok and mean_ok:
        stats['status'] = 'PASS'
    elif (sys_ok or 100 <= sys_mmhg <= 140) and (dia_ok or 50 <= dia_mmhg <= 95) and (mean_ok or 75 <= mean_mmhg <= 110):
        stats['status'] = 'WARN'
    else:
        stats['status'] = 'FAIL'
    
    stats['notes'].append(f"Sys={sys_mmhg:.1f}, Dia={dia_mmhg:.1f}, Mean={mean_mmhg:.1f} mmHg")
    stats['notes'].append(f"Expected: Sys 110-130, Dia 65-85, Mean 90-100 mmHg")
    
    return stats

def estimate_heart_rate(signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Estimate heart rate from pressure peaks."""
    hr = {'bpm': None, 'period_s': None, 'notes': []}
    
    if 'p_start_gauge' not in signals or 'time' not in signals:
        hr['notes'].append("Insufficient data")
        return hr
    
    time = signals['time']
    p = signals['p_start_gauge']
    
    # Find peaks (local maxima)
    if len(p) < 50:
        hr['notes'].append("Too few samples to detect peaks")
        return hr
    
    # Simple peak detection
    peaks = []
    for i in range(1, len(p) - 1):
        if p[i] > p[i-1] and p[i] > p[i+1]:
            peaks.append(i)
    
    if len(peaks) >= 2:
        # Average peak spacing
        peak_times = time[peaks]
        intervals = np.diff(peak_times)
        avg_interval = np.mean(intervals)
        bpm = 60.0 / avg_interval if avg_interval > 0 else None
        
        hr['bpm'] = bpm
        hr['period_s'] = avg_interval
        hr['notes'].append(f"Detected {len(peaks)} peaks, HR ≈ {bpm:.0f} bpm" if bpm else "Could not estimate HR")
    else:
        hr['notes'].append("Fewer than 2 peaks detected")
    
    return hr

# ============================================================================
# 5. WAVEFORM MORPHOLOGY
# ============================================================================

def check_waveform_quality(signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Check aortic pressure waveform morphology (systolic upstroke, diastolic decay)."""
    quality = {'has_systolic_upstroke': False, 'has_diastolic_decay': False, 'notes': [], 'status': 'PASS'}
    
    if 'p_start_gauge' not in signals or 'time' not in signals:
        return quality
    
    p = signals['p_start_gauge']
    if len(p) < 100:
        quality['notes'].append("Insufficient samples")
        return quality
    
    # Last cycle
    samples_per_cycle = max(100, len(p) // 10)
    last_cycle = p[-samples_per_cycle:]
    
    # Check upstroke (first half): dP/dt > 0
    mid = len(last_cycle) // 2
    upstroke_dp = last_cycle[mid] - last_cycle[0]
    downstroke_dp = last_cycle[-1] - last_cycle[mid]
    
    quality['has_systolic_upstroke'] = upstroke_dp > 5  # mmHg rise expected
    quality['has_diastolic_decay'] = downstroke_dp < -5  # mmHg decay expected
    
    if quality['has_systolic_upstroke'] and quality['has_diastolic_decay']:
        quality['notes'].append("✓ Realistic pressure waveform (upstroke & decay)")
        quality['status'] = 'PASS'
    elif quality['has_systolic_upstroke'] or quality['has_diastolic_decay']:
        quality['notes'].append("⚠ Partial waveform morphology")
        quality['status'] = 'WARN'
    else:
        quality['notes'].append("❌ Unrealistic pressure waveform")
        quality['status'] = 'FAIL'
    
    return quality

# ============================================================================
# 6. CIRCLE OF WILLIS PLAUSIBILITY
# ============================================================================

def analyze_cow_vessels() -> Dict[str, Any]:
    """Analyze Circle of Willis vessel flows and pressures."""
    cow_analysis = {
        'vessels': {},
        'asymmetry_l_r': {},
        'flow_zero_count': 0,
        'pressure_discontinuities': [],
        'status': 'PASS',
        'notes': []
    }
    
    # Load CoW vessel data
    for vessel_id, (side, name) in COW_MAPPING.items():
        vessel_data = load_arterial_data(vessel_id)
        if vessel_data is None:
            continue
        
        signals = extract_vessel_signals(vessel_data)
        if len(signals.get('q_start', [])) < 100:
            continue
        
        # Use last cycle
        samples_per_cycle = max(100, len(signals['q_start']) // 10)
        q_mean = np.mean(signals['q_start'][-samples_per_cycle:]) * 60000  # L/min
        p_mean = np.mean(signals.get('p_start_gauge', [0])[-samples_per_cycle:])
        
        cow_analysis['vessels'][f"{side}-{name}" if side else name] = {
            'vessel_id': vessel_id,
            'q_mean_lmin': q_mean,
            'p_mean_mmhg': p_mean,
            'is_zero_flow': abs(q_mean) < 0.01
        }
        
        if abs(q_mean) < 0.01:
            cow_analysis['flow_zero_count'] += 1
    
    # Check L/R asymmetry for paired vessels
    for key in ['MCA', 'A1', 'P1', 'P2', 'ICA']:
        r_key = f"R-{key}"
        l_key = f"L-{key}"
        
        if r_key in cow_analysis['vessels'] and l_key in cow_analysis['vessels']:
            q_r = cow_analysis['vessels'][r_key]['q_mean_lmin']
            q_l = cow_analysis['vessels'][l_key]['q_mean_lmin']
            
            if q_r + q_l > 0.01:
                asymmetry = abs(q_r - q_l) / (q_r + q_l) * 100 if (q_r + q_l) > 0 else 0
                cow_analysis['asymmetry_l_r'][key] = asymmetry
    
    # Check for unrealistic flows in communicating arteries
    for comm_vessel in ['Acom', 'R-Pcom', 'L-Pcom']:
        if comm_vessel in cow_analysis['vessels']:
            q = cow_analysis['vessels'][comm_vessel]['q_mean_lmin']
            if abs(q) > 0.5:  # Communicating arteries shouldn't carry huge flow
                cow_analysis['pressure_discontinuities'].append(f"{comm_vessel}: unusually high flow {q:.3f} L/min")
    
    # Status
    if cow_analysis['flow_zero_count'] > 4:
        cow_analysis['status'] = 'FAIL'
        cow_analysis['notes'].append(f"❌ {cow_analysis['flow_zero_count']} CoW vessels have near-zero flow (unexpected)")
    elif cow_analysis['asymmetry_l_r']:
        max_asym = max(cow_analysis['asymmetry_l_r'].values())
        if max_asym > 50:
            cow_analysis['status'] = 'WARN'
            cow_analysis['notes'].append(f"⚠ Large L/R asymmetry detected (max {max_asym:.1f}%)")
        else:
            cow_analysis['notes'].append(f"✓ CoW L/R asymmetry within range (max {max_asym:.1f}%)")
    else:
        cow_analysis['notes'].append("✓ CoW vessels flowing")
    
    return cow_analysis

# ============================================================================
# 7. PLOTTING
# ============================================================================

def create_plots(aorta_signals: Dict, aorta_id: str = 'A1'):
    """Create validation plots."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Aortic pressure vs time
    ax1 = plt.subplot(3, 3, 1)
    time = aorta_signals['time']
    if 'p_start_gauge' in aorta_signals:
        p = aorta_signals['p_start_gauge']
        ax1.plot(time, p, 'b-', linewidth=0.8)
        ax1.set_ylabel('Pressure (mmHg)', fontsize=10)
        ax1.set_title('Aortic Pressure vs Time', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time (s)')
    
    # Plot 2: Aortic flow vs time
    ax2 = plt.subplot(3, 3, 2)
    if 'q_start' in aorta_signals:
        q = aorta_signals['q_start'] * 60000  # L/min
        ax2.plot(time, q, 'r-', linewidth=0.8)
        ax2.set_ylabel('Flow (L/min)', fontsize=10)
        ax2.set_title('Aortic Flow vs Time', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time (s)')
    
    # Plot 3: Aortic velocity vs time
    ax3 = plt.subplot(3, 3, 3)
    if 'u_start' in aorta_signals:
        u = aorta_signals['u_start']
        ax3.plot(time, u, 'g-', linewidth=0.8)
        ax3.set_ylabel('Velocity (m/s)', fontsize=10)
        ax3.set_title('Aortic Velocity vs Time', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Time (s)')
    
    # Plot 4: Last 5 cycles overlay (pressure)
    ax4 = plt.subplot(3, 3, 4)
    if 'p_start_gauge' in aorta_signals:
        p = aorta_signals['p_start_gauge']
        samples_per_cycle = max(100, len(p) // 10)
        n_cycles = min(5, len(p) // samples_per_cycle)
        
        for i in range(n_cycles):
            start_idx = len(p) - (n_cycles - i) * samples_per_cycle
            end_idx = start_idx + samples_per_cycle
            if end_idx <= len(p):
                cycle_time = np.linspace(0, 1, samples_per_cycle)
                ax4.plot(cycle_time, p[start_idx:end_idx], alpha=0.6, linewidth=1.2, label=f'Cycle {i+1}')
        
        ax4.set_ylabel('Pressure (mmHg)', fontsize=10)
        ax4.set_title('Pressure Cycle Overlay (Last 5)', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Cycle fraction')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Flow cycle overlay
    ax5 = plt.subplot(3, 3, 5)
    if 'q_start' in aorta_signals:
        q = aorta_signals['q_start'] * 60000
        samples_per_cycle = max(100, len(q) // 10)
        n_cycles = min(5, len(q) // samples_per_cycle)
        
        for i in range(n_cycles):
            start_idx = len(q) - (n_cycles - i) * samples_per_cycle
            end_idx = start_idx + samples_per_cycle
            if end_idx <= len(q):
                cycle_time = np.linspace(0, 1, samples_per_cycle)
                ax5.plot(cycle_time, q[start_idx:end_idx], alpha=0.6, linewidth=1.2, label=f'Cycle {i+1}')
        
        ax5.set_ylabel('Flow (L/min)', fontsize=10)
        ax5.set_title('Flow Cycle Overlay (Last 5)', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Cycle fraction')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Pressure vs Flow (loop)
    ax6 = plt.subplot(3, 3, 6)
    if 'p_start_gauge' in aorta_signals and 'q_start' in aorta_signals:
        p_last = aorta_signals['p_start_gauge'][-samples_per_cycle:]
        q_last = aorta_signals['q_start'][-samples_per_cycle:] * 60000
        ax6.plot(q_last, p_last, 'b-', linewidth=1)
        ax6.scatter(q_last[0], p_last[0], color='g', s=50, label='Start', zorder=5)
        ax6.scatter(q_last[-1], p_last[-1], color='r', s=50, label='End', zorder=5)
        ax6.set_xlabel('Flow (L/min)', fontsize=10)
        ax6.set_ylabel('Pressure (mmHg)', fontsize=10)
        ax6.set_title('P-Q Loop (Last Cycle)', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: CoW flows (L vs R MCA if available)
    ax7 = plt.subplot(3, 3, 7)
    cow_analysis = analyze_cow_vessels()
    mca_vessels = ['R-MCA', 'L-MCA']
    mca_flows = []
    mca_labels = []
    
    for vessel in mca_vessels:
        if vessel in cow_analysis['vessels']:
            q = cow_analysis['vessels'][vessel]['q_mean_lmin']
            mca_flows.append(q)
            mca_labels.append(vessel)
    
    if mca_flows:
        colors = ['red', 'blue']
        ax7.bar(mca_labels, mca_flows, color=colors[:len(mca_flows)], alpha=0.7, edgecolor='black')
        ax7.set_ylabel('Mean Flow (L/min)', fontsize=10)
        ax7.set_title('CoW MCA L/R Comparison', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 8: All CoW vessels flow
    ax8 = plt.subplot(3, 3, 8)
    vessel_names = list(cow_analysis['vessels'].keys())[:10]
    vessel_flows = [cow_analysis['vessels'][v]['q_mean_lmin'] for v in vessel_names]
    
    colors_cow = ['green' if q > 0.01 else 'red' for q in vessel_flows]
    ax8.barh(vessel_names, vessel_flows, color=colors_cow, alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Mean Flow (L/min)', fontsize=10)
    ax8.set_title('CoW Vessel Flows', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='x')
    ax8.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 9: Summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = "VALIDATION SUMMARY\n" + "="*35 + "\n"
    if 'p_start_gauge' in aorta_signals:
        p_last_cycle = aorta_signals['p_start_gauge'][-samples_per_cycle:]
        summary_text += f"Aortic Pressure:\n"
        summary_text += f"  Sys: {np.max(p_last_cycle):.1f} mmHg\n"
        summary_text += f"  Dia: {np.min(p_last_cycle):.1f} mmHg\n"
        summary_text += f"  Mean: {np.mean(p_last_cycle):.1f} mmHg\n"
    
    if 'q_start' in aorta_signals:
        q_last_cycle = aorta_signals['q_start'][-samples_per_cycle:] * 60000
        summary_text += f"Cardiac Output:\n"
        summary_text += f"  CO: {np.mean(q_last_cycle):.2f} L/min\n"
    
    summary_text += f"\nCoW Status:\n"
    summary_text += f"  Total vessels: {len(cow_analysis['vessels'])}\n"
    summary_text += f"  Zero-flow: {cow_analysis['flow_zero_count']}\n"
    summary_text += f"  Status: {cow_analysis['status']}\n"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'validation_report_plots.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plots saved to {OUTPUT_DIR / 'validation_report_plots.png'}")
    plt.close()

# ============================================================================
# 8. REPORT GENERATION
# ============================================================================

def generate_report(schema: Dict, aorta_id: str, all_results: Dict):
    """Generate comprehensive validation report."""
    
    report_lines = [
        "=" * 80,
        "FIRSTBLOOD PATIENT-SPECIFIC SIMULATION VALIDATION REPORT",
        "Patient 025 - Circle of Willis Model",
        "=" * 80,
        "",
        f"Simulation Date: 2026-02-04",
        f"Results Directory: {RESULTS_DIR}",
        f"Aortic Vessel ID: {aorta_id}",
        "",
        "=" * 80,
        "1. OUTPUT SCHEMA DISCOVERY",
        "=" * 80,
        "",
        f"Total arterial vessels identified: {len(schema)}",
        "",
        "Vessel classification (sample):",
        "",
    ]
    
    # Sample vessel info
    for i, (vessel_id, info) in enumerate(list(schema.items())[:5]):
        report_lines.append(f"  {vessel_id}: {info['n_cols']} columns, shape={info['shape']}")
    
    report_lines.extend([
        "",
        "Pressure baseline: ABSOLUTE (~100000 Pa = 1 atm)",
        "Pressure conversion: gauge(mmHg) = (absolute_Pa - 100000) / 133.322",
        "",
        "=" * 80,
        "2. NUMERICAL CORRECTNESS",
        "=" * 80,
        "",
    ])
    
    # NaN/Inf check
    nan_inf = all_results['nan_inf']
    report_lines.append(f"NaN/Inf Check:")
    report_lines.append(f"  Status: {'PASS' if nan_inf['ok'] else 'FAIL'}")
    report_lines.append(f"  NaN count: {nan_inf['nan_count']}")
    report_lines.append(f"  Inf count: {nan_inf['inf_count']}")
    if nan_inf['extreme_values']:
        for ev in nan_inf['extreme_values']:
            report_lines.append(f"    - {ev}")
    report_lines.append("")
    
    # Convergence
    convergence = all_results['convergence']
    report_lines.append(f"Periodic Convergence (Last 5 Cycles):")
    report_lines.append(f"  Status: {convergence['status']}")
    report_lines.append(f"  Cycles analyzed: {convergence['cycles_analyzed']}")
    if convergence['rms_error_pct'] is not None:
        report_lines.append(f"  RMS error: {convergence['rms_error_pct']:.4f}%")
    for note in convergence['notes']:
        report_lines.append(f"    {note}")
    report_lines.append("")
    
    # Mass conservation
    conservation = all_results['conservation']
    report_lines.append(f"Mass Conservation:")
    report_lines.append(f"  Status: {conservation['status']}")
    if conservation['co_in_lmin'] is not None:
        report_lines.append(f"  CO_in: {conservation['co_in_lmin']:.2f} L/min")
    if conservation['co_out_lmin'] is not None:
        report_lines.append(f"  CO_out: {conservation['co_out_lmin']:.2f} L/min")
    if conservation['error_pct'] is not None:
        report_lines.append(f"  Mass error: {conservation['error_pct']:.2f}%")
    for note in conservation['notes']:
        report_lines.append(f"    {note}")
    report_lines.append("")
    
    report_lines.extend([
        "=" * 80,
        "3. GLOBAL PHYSIOLOGY",
        "=" * 80,
        "",
    ])
    
    # Cardiac output
    co = all_results['co']
    report_lines.append(f"Cardiac Output:")
    report_lines.append(f"  Status: {co['status']}")
    if co['co_lmin'] is not None:
        report_lines.append(f"  CO: {co['co_lmin']:.2f} L/min")
    for note in co['notes']:
        report_lines.append(f"    {note}")
    report_lines.append("")
    
    # Aortic pressure
    aortic_p = all_results['aortic_pressure']
    report_lines.append(f"Aortic Pressure (Last Cycle):")
    report_lines.append(f"  Status: {aortic_p['status']}")
    report_lines.append(f"  Systolic: {aortic_p['systolic_mmhg']:.1f} mmHg (target 110-130)")
    report_lines.append(f"  Diastolic: {aortic_p['diastolic_mmhg']:.1f} mmHg (target 65-85)")
    report_lines.append(f"  Mean: {aortic_p['mean_mmhg']:.1f} mmHg (target 90-100)")
    for note in aortic_p['notes']:
        report_lines.append(f"    {note}")
    report_lines.append("")
    
    # Heart rate
    hr = all_results['hr']
    report_lines.append(f"Heart Rate:")
    if hr['bpm'] is not None:
        report_lines.append(f"  Estimated HR: {hr['bpm']:.0f} bpm")
        report_lines.append(f"  Cycle period: {hr['period_s']:.3f} s")
    for note in hr['notes']:
        report_lines.append(f"    {note}")
    report_lines.append("")
    
    report_lines.extend([
        "=" * 80,
        "4. WAVEFORM MORPHOLOGY",
        "=" * 80,
        "",
    ])
    
    # Waveform quality
    waveform = all_results['waveform']
    report_lines.append(f"Aortic Pressure Waveform:")
    report_lines.append(f"  Status: {waveform['status']}")
    report_lines.append(f"  Systolic upstroke: {'Yes' if waveform['has_systolic_upstroke'] else 'No'}")
    report_lines.append(f"  Diastolic decay: {'Yes' if waveform['has_diastolic_decay'] else 'No'}")
    for note in waveform['notes']:
        report_lines.append(f"    {note}")
    report_lines.append("")
    
    report_lines.extend([
        "=" * 80,
        "5. CIRCLE OF WILLIS PLAUSIBILITY",
        "=" * 80,
        "",
    ])
    
    # CoW analysis
    cow = all_results['cow']
    report_lines.append(f"CoW Vessel Status:")
    report_lines.append(f"  Overall Status: {cow['status']}")
    report_lines.append(f"  Total CoW vessels: {len(cow['vessels'])}")
    report_lines.append(f"  Zero-flow vessels: {cow['flow_zero_count']}")
    report_lines.append("")
    
    report_lines.append(f"Vessel-by-Vessel Flow (L/min):")
    for vessel, info in sorted(cow['vessels'].items()):
        status = "✓" if info['q_mean_lmin'] > 0.01 else "✗"
        report_lines.append(f"  {status} {vessel:<12} Q_mean={info['q_mean_lmin']:>8.3f}, P_mean={info['p_mean_mmhg']:>7.1f} mmHg")
    report_lines.append("")
    
    if cow['asymmetry_l_r']:
        report_lines.append(f"L/R Asymmetry Analysis:")
        for vessel, asym in sorted(cow['asymmetry_l_r'].items()):
            report_lines.append(f"  {vessel}: {asym:.1f}% asymmetry")
        report_lines.append("")
    
    for note in cow['notes']:
        report_lines.append(f"  {note}")
    report_lines.append("")
    
    if cow['pressure_discontinuities']:
        report_lines.append("Pressure Anomalies:")
        for anom in cow['pressure_discontinuities']:
            report_lines.append(f"  ⚠ {anom}")
        report_lines.append("")
    
    report_lines.extend([
        "=" * 80,
        "6. FINAL VERDICT",
        "=" * 80,
        "",
    ])
    
    # Overall status
    statuses = [
        nan_inf['ok'],
        convergence['status'] in ['PASS', 'WARN'],
        aortic_p['status'] in ['PASS', 'WARN'],
        co['status'] in ['PASS', 'WARN'],
        cow['status'] in ['PASS', 'WARN']
    ]
    
    if all(statuses):
        overall = "✓ PASS - Simulation is numerically stable and biologically plausible"
    elif sum(statuses) >= 3:
        overall = "⚠ WARN - Simulation is mostly valid but has some warnings"
    else:
        overall = "✗ FAIL - Simulation has significant issues"
    
    report_lines.append(overall)
    report_lines.append("")
    report_lines.append("CHECKLIST:")
    report_lines.append(f"  [{'X' if nan_inf['ok'] else ' '}] Numerical stability (no NaN/Inf)")
    report_lines.append(f"  [{'X' if convergence['status'] in ['PASS', 'WARN'] else ' '}] Periodic convergence")
    report_lines.append(f"  [{'X' if aortic_p['status'] in ['PASS', 'WARN'] else ' '}] Aortic pressure in range")
    report_lines.append(f"  [{'X' if co['status'] in ['PASS', 'WARN'] else ' '}] Cardiac output in range")
    report_lines.append(f"  [{'X' if waveform['status'] in ['PASS', 'WARN'] else ' '}] Realistic waveform morphology")
    report_lines.append(f"  [{'X' if cow['status'] in ['PASS', 'WARN'] else ' '}] CoW plausibility")
    report_lines.append("")
    
    return "\n".join(report_lines)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("FIRSTBLOOD PATIENT-SPECIFIC VALIDATION WORKFLOW")
    print("=" * 80)
    print()
    
    # Step 0: Schema discovery
    print("📊 Step 0: Discovering output schema...")
    schema = discover_arterial_schema()
    print(f"  ✓ Discovered {len(schema)} arterial vessels")
    
    # Identify aorta
    print("\n🔍 Identifying aortic vessel...")
    aorta_id = identify_aorta(schema)
    print(f"  ✓ Aortic vessel: {aorta_id}")
    
    # Load aortic data
    print(f"\n📈 Loading aortic signals from {aorta_id}...")
    aorta_data = load_arterial_data(aorta_id)
    if aorta_data is None:
        print("  ❌ Failed to load aortic data")
        return
    
    aorta_signals = extract_vessel_signals(aorta_data)
    print(f"  ✓ Loaded {len(aorta_signals)} signal types")
    
    # Load heart data if available
    heart_data_file = RESULTS_DIR / "heart_kim_lit" / "L_lv_aorta.txt"
    heart_signals = {}
    if heart_data_file.exists():
        heart_data = np.loadtxt(heart_data_file, delimiter=',')
        if heart_data.ndim == 1:
            heart_data = heart_data.reshape(1, -1)
        heart_signals['q_lv'] = heart_data[:, 1] if heart_data.shape[1] > 1 else np.array([])
    
    # Step 1: Numerical correctness
    print("\n🔢 Step 1: Checking numerical correctness...")
    nan_inf = check_nan_inf(aorta_signals)
    print(f"  {'✓' if nan_inf['ok'] else '✗'} NaN/Inf check: {nan_inf['ok']}")
    
    convergence = compute_convergence(aorta_signals)
    print(f"  {convergence['status']} Convergence: {convergence['rms_error_pct']:.4f}%" if convergence['rms_error_pct'] else f"  {convergence['status']} Convergence: {convergence['notes']}")
    
    conservation = check_mass_conservation(aorta_signals, heart_signals)
    print(f"  {conservation['status']} Mass conservation: {conservation['error_pct']:.2f}%" if conservation['error_pct'] else f"  {conservation['status']} Mass conservation")
    
    # Step 2: Global physiology
    print("\n💓 Step 2: Computing global physiology...")
    co = compute_cardiac_output(aorta_signals)
    print(f"  {co['status']} Cardiac output: {co['co_lmin']:.2f} L/min" if co['co_lmin'] else f"  {co['status']} Cardiac output")
    
    aortic_p = compute_aortic_pressure_stats(aorta_signals)
    print(f"  {aortic_p['status']} Aortic pressure: {aortic_p['systolic_mmhg']:.1f}/{aortic_p['diastolic_mmhg']:.1f}/{aortic_p['mean_mmhg']:.1f} mmHg" if aortic_p['systolic_mmhg'] else f"  {aortic_p['status']} Aortic pressure")
    
    hr = estimate_heart_rate(aorta_signals)
    print(f"  ℹ️  Heart rate: {hr['bpm']:.0f} bpm" if hr['bpm'] else f"  ℹ️  Heart rate: unknown")
    
    # Step 3: Waveform morphology
    print("\n📉 Step 3: Checking waveform morphology...")
    waveform = check_waveform_quality(aorta_signals)
    print(f"  {waveform['status']} Pressure waveform: {', '.join(waveform['notes'])}")
    
    # Step 4: CoW plausibility
    print("\n🧠 Step 4: Circle of Willis analysis...")
    cow = analyze_cow_vessels()
    print(f"  {cow['status']} CoW status: {len(cow['vessels'])} vessels, {cow['flow_zero_count']} zero-flow")
    for note in cow['notes']:
        print(f"      {note}")
    
    # Step 5: Plotting
    print("\n📊 Step 5: Creating plots...")
    create_plots(aorta_signals, aorta_id)
    
    # Step 6: Generate report
    print("\n📝 Step 6: Generating validation report...")
    all_results = {
        'nan_inf': nan_inf,
        'convergence': convergence,
        'conservation': conservation,
        'co': co,
        'aortic_pressure': aortic_p,
        'hr': hr,
        'waveform': waveform,
        'cow': cow
    }
    
    report = generate_report(schema, aorta_id, all_results)
    
    # Save report
    report_file = OUTPUT_DIR / 'validation_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"  ✓ Report saved to {report_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(report)
    print("\n📁 Output files:")
    print(f"   - {report_file}")
    print(f"   - {OUTPUT_DIR / 'validation_report_plots.png'}")
    print()

if __name__ == "__main__":
    main()
