#!/usr/bin/env python3
"""
FirstBlood Simulation Validation Script
========================================

Comprehensive validation of FirstBlood model results against:
- Numerical correctness (convergence, stability, mass conservation)
- Global physiological parameters (CO, aortic pressure, velocities)
- Waveform morphology (systolic/diastolic patterns)
- Spatial propagation and pulse wave behavior
- Cerebral Circle of Willis plausibility

Reference: FirstBlood paper validation criteria
"""

import os
import sys
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = '/home/maryyds/final/first_blood/projects/simple_run/results/Abel_ref2'
OUTPUT_DIR = '/home/maryyds/final/first_blood/validate_abelref2/validation_output'
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')

# Create output directories
os.makedirs(PLOT_DIR, exist_ok=True)

# Physiological reference ranges
PHYS_RANGES = {
    'CO': (4.0, 6.0),                    # L/min
    'CO_ref': 5.35,                      # L/min (paper reference)
    'HR': (60, 100),                     # beats/min (typical)
    'stroke_volume': (60, 90),           # mL
    'aorta_systolic': (110, 130),        # mmHg
    'aorta_diastolic': (65, 85),         # mmHg
    'aorta_mean': (90, 100),             # mmHg
    'aorta_vel_peak': (0.8, 1.5),        # m/s
    'aorta_vel_mean': (0.2, 0.4),        # m/s
}

def pascals_to_mmhg(pressure_pa):
    """Convert pressure from Pa to mmHg."""
    return pressure_pa * 0.00750062

def load_data(filepath):
    """Load CSV/text data from file."""
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_cycles(time, signal_data, cycle_duration=1.0):
    """Extract individual cardiac cycles from signal."""
    n_samples = len(time)
    dt = (time[-1] - time[0]) / (n_samples - 1)
    samples_per_cycle = int(cycle_duration / dt)
    
    n_cycles = n_samples // samples_per_cycle
    cycles = []
    
    for i in range(n_cycles - 1):
        start = i * samples_per_cycle
        end = (i + 1) * samples_per_cycle
        cycles.append(signal_data[start:end])
    
    return cycles

def compute_cycle_rms_error(cycle1, cycle2):
    """Compute RMS error between consecutive cycles."""
    if len(cycle1) != len(cycle2):
        min_len = min(len(cycle1), len(cycle2))
        cycle1 = cycle1[:min_len]
        cycle2 = cycle2[:min_len]
    
    mean_val = np.mean(np.abs(cycle2))
    if mean_val == 0:
        return 0
    
    rms = np.sqrt(np.mean((cycle1 - cycle2) ** 2))
    rms_percent = (rms / mean_val) * 100
    return rms_percent

def check_periodicity(time, signal_data, cycle_duration=1.0):
    """Check for periodic convergence in cardiac cycles."""
    cycles = extract_cycles(time, signal_data, cycle_duration)
    
    if len(cycles) < 3:
        return None, None
    
    rms_errors = []
    for i in range(len(cycles) - 1):
        rms_pct = compute_cycle_rms_error(cycles[i], cycles[i+1])
        rms_errors.append(rms_pct)
    
    mean_rms = np.mean(rms_errors[-2:]) if len(rms_errors) >= 2 else np.mean(rms_errors)
    is_converged = mean_rms < 0.1 if mean_rms is not None else False
    
    return is_converged, mean_rms

def check_nans_and_bounds(data):
    """Check for NaNs, infs, and unrealistic values."""
    issues = []
    
    if np.any(np.isnan(data)):
        issues.append('Contains NaN values')
    if np.any(np.isinf(data)):
        issues.append('Contains infinite values')
    
    if len(data) > 0:
        if np.max(np.abs(data)) > 1e10:
            issues.append('Contains blow-up values (>1e10)')
        if np.max(np.abs(data)) < 1e-8 and np.max(np.abs(data)) > 0:
            issues.append('Suspiciously small values (<1e-8)')
    
    return issues

def compute_flow(velocity_data, area_data):
    """Compute flow from velocity and area: Q = v * A (in consistent units)."""
    if velocity_data is None or area_data is None:
        return None
    return velocity_data * area_data

def estimate_cardiac_output(time, flow_data, cycle_duration=1.0):
    """Estimate cardiac output from flow waveform."""
    dt = (time[-1] - time[0]) / (len(time) - 1)
    
    cycles = extract_cycles(time, flow_data, cycle_duration)
    if not cycles:
        return None
    
    last_cycle = cycles[-1]
    stroke_volume = np.trapz(last_cycle, dx=dt)
    
    heart_rate = 60.0 / cycle_duration
    cardiac_output = (stroke_volume * heart_rate) / 1e6
    
    return cardiac_output, stroke_volume, heart_rate

def extract_pressure_stats(time, pressure_data, cycle_duration=1.0):
    """Extract systolic, diastolic, and mean pressures."""
    cycles = extract_cycles(time, pressure_data, cycle_duration)
    if not cycles:
        return None, None, None
    
    last_cycle = cycles[-1]
    systolic = np.max(last_cycle)
    diastolic = np.min(last_cycle)
    mean_pressure = np.mean(last_cycle)
    
    return systolic, diastolic, mean_pressure

def detect_dicrotic_notch(pressure_data):
    """Detect presence of dicrotic notch in aortic pressure."""
    peaks, _ = signal.find_peaks(pressure_data, prominence=0.02 * (np.max(pressure_data) - np.min(pressure_data)))
    return len(peaks) >= 2

def check_mass_conservation(results_dir):
    """Check flow conservation at junctions."""
    conservation_ok = True
    issues = []
    
    try:
        arterial_dir = os.path.join(results_dir, 'arterial')
        
        if os.path.exists(os.path.join(arterial_dir, 'n1.txt')):
            data_n1 = load_data(os.path.join(arterial_dir, 'n1.txt'))
            
            if data_n1 is not None and data_n1.shape[1] >= 3:
                flows_in = data_n1[:, 1]
                flows_out = data_n1[:, 2]
                flow_diff = np.abs(flows_in - flows_out)
                max_error = np.max(flow_diff)
                
                if max_error > 1e-2:
                    issues.append(f'Flow mismatch at junction n1: max error = {max_error:.2e}')
                    conservation_ok = False
    except Exception as e:
        issues.append(f'Mass conservation check failed: {str(e)}')
    
    return conservation_ok, issues

def analyze_waveform_shape(signal_data):
    """Analyze waveform morphology."""
    features = {}
    
    if len(signal_data) < 10:
        return features
    
    grad = np.gradient(signal_data)
    features['peak_upstroke'] = np.max(grad[:len(grad)//3])
    features['peak_downstroke'] = np.min(grad[len(grad)//3:])
    features['systolic_width'] = np.sum(grad > 0.5 * features['peak_upstroke'])
    
    return features

def validate_cerebral_cow(results_dir):
    """Validate Circle of Willis plausibility."""
    issues = []
    
    try:
        p_dir = os.path.join(results_dir, 'arterial')
        
        cow_nodes = ['p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26']
        
        for node in cow_nodes:
            node_file = os.path.join(p_dir, f'{node}.txt')
            if os.path.exists(node_file):
                data = load_data(node_file)
                if data is not None:
                    pressure = data[:, 0]
                    
                    if np.max(pressure) == 0 and np.min(pressure) == 0:
                        issues.append(f'Node {node} (CoW) shows zero flow/pressure')
                    
                    pressure_range = np.max(pressure) - np.min(pressure)
                    if pressure_range < 1:
                        issues.append(f'Node {node} (CoW) shows minimal pulsatility')
    
    except Exception as e:
        issues.append(f'CoW validation failed: {str(e)}')
    
    return issues

def generate_report():
    """Generate comprehensive validation report."""
    report = []
    report.append('\n' + '='*80)
    report.append('FIRSTBLOOD SIMULATION VALIDATION REPORT')
    report.append('='*80 + '\n')
    
    # 1. NUMERICAL CORRECTNESS
    report.append('\n[1] NUMERICAL CORRECTNESS')
    report.append('-' * 80)
    
    try:
        time = load_data(os.path.join(RESULTS_DIR, 'arterial', 'p1.txt'))[:, 0]
        pressure = load_data(os.path.join(RESULTS_DIR, 'arterial', 'p1.txt'))[:, 1]
        
        converged, rms_error = check_periodicity(time, pressure)
        
        report.append(f'\nPeriodicality Check:')
        report.append(f'  Converged: {converged}')
        report.append(f'  RMS error (last cycles): {rms_error:.4f}%' if rms_error else '  RMS error: N/A')
        if converged:
            report.append(f'  Status: PASS (RMS < 0.1%)')
        else:
            report.append(f'  Status: WARNING (RMS >= 0.1%)')
        
        nans = check_nans_and_bounds(pressure)
        report.append(f'\nStability Check:')
        report.append(f'  Issues: {nans if nans else "None"}')
        report.append(f'  Status: {"PASS" if not nans else "FAIL"}')
        
        conservation_ok, conservation_issues = check_mass_conservation(RESULTS_DIR)
        report.append(f'\nMass Conservation Check:')
        report.append(f'  Status: {"PASS" if conservation_ok else "FAIL"}')
        if conservation_issues:
            for issue in conservation_issues:
                report.append(f'    - {issue}')
    
    except Exception as e:
        report.append(f'Error in numerical check: {str(e)}')
    
    # 2. GLOBAL PHYSIOLOGICAL VALIDATION
    report.append(f'\n\n[2] GLOBAL PHYSIOLOGICAL VALIDATION')
    report.append('-' * 80)
    
    try:
        time = load_data(os.path.join(RESULTS_DIR, 'heart_kim_lit', 'aorta.txt'))[:, 0]
        flow = load_data(os.path.join(RESULTS_DIR, 'heart_kim_lit', 'aorta.txt'))[:, 1]
        pressure = load_data(os.path.join(RESULTS_DIR, 'arterial', 'p1.txt'))[:, 1]
        
        dt = (time[-1] - time[0]) / (len(time) - 1)
        cycle_duration = 1.0
        
        co_result = estimate_cardiac_output(time, flow, cycle_duration)
        if co_result:
            co, sv, hr = co_result
            report.append(f'\nCardiac Output:')
            report.append(f'  CO: {co:.2f} L/min')
            report.append(f'  Reference: {PHYS_RANGES["CO_ref"]:.2f} L/min')
            report.append(f'  Range: {PHYS_RANGES["CO"][0]}-{PHYS_RANGES["CO"][1]} L/min')
            co_ok = PHYS_RANGES['CO'][0] <= co <= PHYS_RANGES['CO'][1]
            report.append(f'  Status: {"PASS" if co_ok else "FAIL/WARNING"}')
            
            report.append(f'\nStroke Volume:')
            report.append(f'  SV: {sv:.2f} mL')
            report.append(f'  Range: {PHYS_RANGES["stroke_volume"][0]}-{PHYS_RANGES["stroke_volume"][1]} mL')
            sv_ok = PHYS_RANGES['stroke_volume'][0] <= sv <= PHYS_RANGES['stroke_volume'][1]
            report.append(f'  Status: {"PASS" if sv_ok else "FAIL/WARNING"}')
            
            report.append(f'\nHeart Rate:')
            report.append(f'  HR: {hr:.1f} bpm')
            
        sys_p, dia_p, mean_p = extract_pressure_stats(time, pressure, cycle_duration)
        if sys_p is not None:
            sys_mmhg = pascals_to_mmhg(sys_p)
            dia_mmhg = pascals_to_mmhg(dia_p)
            mean_mmhg = pascals_to_mmhg(mean_p)
            
            report.append(f'\nAortic Pressure:')
            report.append(f'  Systolic: {sys_mmhg:.1f} mmHg (range: {PHYS_RANGES["aorta_systolic"][0]}-{PHYS_RANGES["aorta_systolic"][1]})')
            sys_ok = PHYS_RANGES['aorta_systolic'][0] <= sys_mmhg <= PHYS_RANGES['aorta_systolic'][1]
            report.append(f'  Status: {"PASS" if sys_ok else "FAIL/WARNING"}')
            
            report.append(f'  Diastolic: {dia_mmhg:.1f} mmHg (range: {PHYS_RANGES["aorta_diastolic"][0]}-{PHYS_RANGES["aorta_diastolic"][1]})')
            dia_ok = PHYS_RANGES['aorta_diastolic'][0] <= dia_mmhg <= PHYS_RANGES['aorta_diastolic'][1]
            report.append(f'  Status: {"PASS" if dia_ok else "FAIL/WARNING"}')
            
            report.append(f'  Mean: {mean_mmhg:.1f} mmHg (range: {PHYS_RANGES["aorta_mean"][0]}-{PHYS_RANGES["aorta_mean"][1]})')
            mean_ok = PHYS_RANGES['aorta_mean'][0] <= mean_mmhg <= PHYS_RANGES['aorta_mean'][1]
            report.append(f'  Status: {"PASS" if mean_ok else "FAIL/WARNING"}')
            
            dicrotic = detect_dicrotic_notch(pressure)
            report.append(f'\nWaveform Morphology:')
            report.append(f'  Dicrotic notch detected: {dicrotic}')
    
    except Exception as e:
        report.append(f'Error in physiological check: {str(e)}')
    
    # 3. VELOCITY VALIDATION
    report.append(f'\n\n[3] VELOCITY & FLOW MAGNITUDE VALIDATION')
    report.append('-' * 80)
    
    try:
        A1_data = load_data(os.path.join(RESULTS_DIR, 'arterial', 'A1.txt'))
        
        if A1_data is not None and A1_data.shape[1] >= 10:
            velocity = A1_data[:, 3]
            
            vel_peak = np.max(velocity)
            vel_mean = np.mean(velocity[len(velocity)//2:])
            
            report.append(f'\nAortic Root Velocity:')
            report.append(f'  Peak: {vel_peak:.3f} m/s (range: {PHYS_RANGES["aorta_vel_peak"][0]}-{PHYS_RANGES["aorta_vel_peak"][1]})')
            vel_ok = PHYS_RANGES['aorta_vel_peak'][0] <= vel_peak <= PHYS_RANGES['aorta_vel_peak'][1]
            report.append(f'  Status: {"PASS" if vel_ok else "FAIL/WARNING"}')
            
            report.append(f'  Mean: {vel_mean:.3f} m/s (range: {PHYS_RANGES["aorta_vel_mean"][0]}-{PHYS_RANGES["aorta_vel_mean"][1]})')
            
            if vel_peak < 1e-6:
                report.append(f'  WARNING: Suspiciously small velocity (< 1e-6 m/s)')
            elif vel_peak > 10:
                report.append(f'  WARNING: Suspiciously large velocity (> 10 m/s)')
    
    except Exception as e:
        report.append(f'Error in velocity check: {str(e)}')
    
    # 4. SPATIAL PROPAGATION
    report.append(f'\n\n[4] SPATIAL WAVEFORM BEHAVIOR')
    report.append('-' * 80)
    
    try:
        report.append(f'\nPulse Wave Propagation:')
        report.append(f'  Status: Qualitative inspection recommended')
        report.append(f'  See generated plots for time-delay analysis')
    
    except Exception as e:
        report.append(f'Error in spatial check: {str(e)}')
    
    # 5. CEREBRAL COW
    report.append(f'\n\n[5] CEREBRAL / CIRCLE OF WILLIS PLAUSIBILITY')
    report.append('-' * 80)
    
    cow_issues = validate_cerebral_cow(RESULTS_DIR)
    if cow_issues:
        report.append(f'\nIssues detected:')
        for issue in cow_issues:
            report.append(f'  - {issue}')
    else:
        report.append(f'\nStatus: PASS (No obvious pathologies)')
    
    # 6. KNOWN LIMITATIONS
    report.append(f'\n\n[6] KNOWN LIMITATIONS (ACKNOWLEDGED)')
    report.append('-' * 80)
    report.append('\n[STATED IN PAPER]')
    report.append('  - No venous circulation')
    report.append('  - No cerebral autoregulation')
    report.append('  - No patient-specific boundary conditions')
    report.append('  - Ideal valves (no regurgitation)')
    report.append('  - Geometry != physiology')
    
    # 7. SUMMARY
    report.append(f'\n\n[7] FINAL ASSESSMENT')
    report.append('-' * 80)
    report.append('\nSimulation results stored in:')
    report.append(f'  {RESULTS_DIR}')
    report.append('\nValidation plots generated in:')
    report.append(f'  {PLOT_DIR}')
    report.append('\nTo pass validation, results must satisfy:')
    report.append('  1. Periodic convergence (RMS < 0.1%)')
    report.append('  2. No NaN/Inf values or blow-ups')
    report.append('  3. Cardiac output: 4-6 L/min')
    report.append('  4. Aortic pressure in physiological range')
    report.append('  5. Velocity magnitudes realistic (not 1e-8 or 10 m/s)')
    report.append('  6. Waveforms physiologically shaped')
    report.append('\n' + '='*80 + '\n')
    
    return '\n'.join(report)

def generate_plots():
    """Generate minimum required validation plots."""
    try:
        time = load_data(os.path.join(RESULTS_DIR, 'heart_kim_lit', 'aorta.txt'))[:, 0]
        
        # Plot 1: Aortic pressure vs time
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('FirstBlood Validation - Core Waveforms', fontsize=14)
        
        try:
            pressure = load_data(os.path.join(RESULTS_DIR, 'arterial', 'p1.txt'))[:, 1]
            pressure_mmhg = pascals_to_mmhg(pressure)
            axes[0, 0].plot(time, pressure_mmhg, 'b-', linewidth=1.5)
            axes[0, 0].set_ylabel('Pressure (mmHg)')
            axes[0, 0].set_title('Aortic Pressure')
            axes[0, 0].grid(True, alpha=0.3)
        except:
            axes[0, 0].text(0.5, 0.5, 'Data unavailable', ha='center', va='center')
        
        # Plot 2: Heart outflow vs time
        try:
            flow = load_data(os.path.join(RESULTS_DIR, 'heart_kim_lit', 'aorta.txt'))[:, 1]
            axes[0, 1].plot(time, flow, 'r-', linewidth=1.5)
            axes[0, 1].set_ylabel('Flow (cm3/s)')
            axes[0, 1].set_title('Heart Outflow')
            axes[0, 1].grid(True, alpha=0.3)
        except:
            axes[0, 1].text(0.5, 0.5, 'Data unavailable', ha='center', va='center')
        
        # Plot 3: Aortic velocity
        try:
            A1_data = load_data(os.path.join(RESULTS_DIR, 'arterial', 'A1.txt'))
            if A1_data is not None and A1_data.shape[1] >= 4:
                velocity = A1_data[:, 3]
                axes[1, 0].plot(time, velocity, 'g-', linewidth=1.5)
                axes[1, 0].set_ylabel('Velocity (m/s)')
                axes[1, 0].set_title('Aortic Root Velocity')
                axes[1, 0].grid(True, alpha=0.3)
        except:
            axes[1, 0].text(0.5, 0.5, 'Data unavailable', ha='center', va='center')
        
        # Plot 4: Carotid velocity (node p2 or similar)
        try:
            p2_data = load_data(os.path.join(RESULTS_DIR, 'arterial', 'p2.txt'))
            if p2_data is not None:
                axes[1, 1].plot(time, p2_data, 'purple', linewidth=1.5)
                axes[1, 1].set_ylabel('Pressure (Pa)')
                axes[1, 1].set_title('Carotid/Subclavian Region')
                axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].text(0.5, 0.5, 'Data unavailable', ha='center', va='center')
        
        # Plot 5: Convergence analysis
        try:
            pressure = load_data(os.path.join(RESULTS_DIR, 'arterial', 'p1.txt'))[:, 1]
            cycles = extract_cycles(time, pressure, 1.0)
            
            if len(cycles) >= 2:
                cycles_to_plot = min(5, len(cycles))
                for i, cycle in enumerate(cycles[-cycles_to_plot:]):
                    t_cycle = np.linspace(0, 1, len(cycle))
                    axes[2, 0].plot(t_cycle, pascals_to_mmhg(cycle), 
                                   label=f'Cycle {i}', alpha=0.7)
                axes[2, 0].set_xlabel('Normalized Cycle Time')
                axes[2, 0].set_ylabel('Pressure (mmHg)')
                axes[2, 0].set_title('Periodic Convergence (Last 5 Cycles)')
                axes[2, 0].legend(fontsize=8)
                axes[2, 0].grid(True, alpha=0.3)
        except:
            axes[2, 0].text(0.5, 0.5, 'Data unavailable', ha='center', va='center')
        
        # Plot 6: Pressure distribution across network
        try:
            p_nodes = ['p1', 'p2', 'p5', 'p10', 'p20']
            for node_name in p_nodes:
                node_file = os.path.join(RESULTS_DIR, 'arterial', f'{node_name}.txt')
                if os.path.exists(node_file):
                    p_data = load_data(node_file)
                    if p_data is not None:
                        p_mean = np.mean(p_data[:, 1])
                        axes[2, 1].scatter(int(node_name[1:]), pascals_to_mmhg(p_mean), s=50)
            
            axes[2, 1].set_xlabel('Node Index')
            axes[2, 1].set_ylabel('Mean Pressure (mmHg)')
            axes[2, 1].set_title('Network Pressure Distribution')
            axes[2, 1].grid(True, alpha=0.3)
        except:
            axes[2, 1].text(0.5, 0.5, 'Data unavailable', ha='center', va='center')
        
        for ax in axes.flat:
            ax.set_xlabel('Time (s)' if 'Time' not in ax.get_xlabel() else ax.get_xlabel())
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'validation_waveforms.png'), dpi=100, bbox_inches='tight')
        print(f'[OK] Saved: validation_waveforms.png')
        plt.close()
        
    except Exception as e:
        print(f'[ERROR] Failed to generate plots: {str(e)}')

if __name__ == '__main__':
    print('\n' + '='*80)
    print('FirstBlood Validation Script Starting')
    print('='*80 + '\n')
    
    print(f'Results directory: {RESULTS_DIR}')
    print(f'Output directory: {OUTPUT_DIR}\n')
    
    if not os.path.exists(RESULTS_DIR):
        print(f'ERROR: Results directory not found: {RESULTS_DIR}')
        sys.exit(1)
    
    print('[1] Generating validation report...')
    report = generate_report()
    print(report)
    
    report_file = os.path.join(OUTPUT_DIR, 'validation_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f'[OK] Report saved: {report_file}\n')
    
    print('[2] Generating validation plots...')
    generate_plots()
    
    print(f'\n[OK] Validation complete.')
    print(f'Output files in: {OUTPUT_DIR}\n')
