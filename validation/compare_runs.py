#!/usr/bin/env python3
"""
FirstBlood Validation: Compare Two Simulation Runs

Compare baseline and patient simulations with comprehensive cardiac metrics:
- Heart rate, systolic/diastolic/MAP pressures
- Stroke volume and cardiac output
- Convergence metrics (periodicity)
- Cerebral flow distribution

Usage:
    python -m validation.compare_runs --patient patient_025 --baseline Abel_ref2
    python -m validation.compare_runs --patient_dir <path> --baseline_dir <path>

Output:
    - <patient_dir>/validation/validation_summary.txt
    - <patient_dir>/validation/validation.json
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

warnings.filterwarnings('ignore')

# Physical constants
PA_TO_MMHG = 133.322
P_ATMO = 1.0e5

# ============================================================================
# Vessel Mapping: Network topology and Circle of Willis structure
# ============================================================================
# Key arterial segments from FirstBlood mesh
COW_VESSELS = {
    # Internal Carotid Arteries (entry to CoW)
    'A12': ('R', 'ICA'),
    'A16': ('L', 'ICA'),
    
    # Anterior Cerebral Arteries (A1 segments - proximal)
    'A68': ('R', 'A1'),
    'A69': ('L', 'A1'),
    
    # Middle Cerebral Arteries (main trunk)
    'A70': ('R', 'MCA'),
    'A73': ('L', 'MCA'),
    
    # Posterior Cerebral Arteries (P1 segments - proximal)
    'A76': ('R', 'P1'),
    'A78': ('L', 'P1'),
    
    # Posterior Cerebral Arteries (P2 segments)
    'A60': ('R', 'P2'),
    'A61': ('L', 'P2'),
    
    # Posterior communicating arteries
    'A62': ('R', 'Pcom'),
    'A63': ('L', 'Pcom'),
    
    # Communicating arteries
    'A77': (None, 'Acom'),  # Anterior communicating
    'A56': (None, 'BA2'),   # Basilar artery (posterior)
    'A59': (None, 'BA1'),   # Basilar artery (anterior)
}

# System arteries (for reference)
AORTA_ID = 'A1'  # Ascending aorta
HEART_ID = 'H'   # Heart


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class CardiacMetrics:
    """Cardiac hemodynamic metrics for a single run"""
    heart_rate: float  # beats per minute
    cycle_period: float  # seconds
    sys_pressure: float  # mmHg
    dia_pressure: float  # mmHg
    map_pressure: float  # mmHg
    stroke_volume: float  # mL
    cardiac_output: float  # L/min
    convergence_rms: float  # RMS diff between last 2 cycles (%)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CerebralFlow:
    """Cerebral autoregulation metrics"""
    total_flow: float  # mL/min
    left_flow: float  # mL/min
    right_flow: float  # mL/min
    left_right_ratio: float
    branches: Dict[str, float]  # Per-branch mean flows
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Complete comparison between baseline and patient"""
    timestamp: str
    baseline_name: str
    patient_name: str
    baseline_metrics: CardiacMetrics
    patient_metrics: CardiacMetrics
    baseline_cerebral: CerebralFlow
    patient_cerebral: CerebralFlow
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'baseline_name': self.baseline_name,
            'patient_name': self.patient_name,
            'baseline_metrics': self.baseline_metrics.to_dict(),
            'patient_metrics': self.patient_metrics.to_dict(),
            'baseline_cerebral': self.baseline_cerebral.to_dict(),
            'patient_cerebral': self.patient_cerebral.to_dict(),
        }


# ============================================================================
# Utilities
# ============================================================================
def detect_delimiter(filepath: Path) -> str:
    """Auto-detect CSV delimiter: comma, semicolon, or whitespace"""
    with open(filepath, 'r') as f:
        line = f.readline().strip()
    
    if ',' in line:
        return ','
    elif ';' in line:
        return ';'
    else:
        return r'\s+'


def load_timeseries(filepath: Path) -> Optional[np.ndarray]:
    """Load numeric data from file, auto-detecting delimiter"""
    if not filepath.exists():
        return None
    
    try:
        delimiter = detect_delimiter(filepath)
        data = np.loadtxt(filepath, delimiter=delimiter)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    except Exception as e:
        print(f"  ⚠ Error loading {filepath.name}: {e}")
        return None


def pressure_to_mmhg(pressure_pa: np.ndarray) -> np.ndarray:
    """Convert Pascal pressure to gauge mmHg"""
    return (pressure_pa - P_ATMO) / PA_TO_MMHG


def find_cycles(time: np.ndarray, signal: np.ndarray, 
                min_cycle_len: int = 100) -> Tuple[List[slice], float]:
    """
    Detect cardiac cycles from signal oscillations.
    Returns: list of cycle slices (start:end indices), and estimated cycle period
    """
    # Find local minima in pressure signal (corresponds to diastolic pressure)
    from scipy import signal as sp_signal
    
    peaks, _ = sp_signal.find_peaks(-signal)  # Find minima
    
    if len(peaks) < 2:
        # Fallback: assume single cycle equal to total duration
        return [slice(0, len(signal))], time[-1] - time[0]
    
    # Calculate average cycle period from peak spacing
    cycle_indices = np.diff(peaks)
    if len(cycle_indices) > 0:
        cycle_period = np.mean(cycle_indices) * (time[1] - time[0])
    else:
        cycle_period = time[-1] - time[0]
    
    # Create cycle slices for last two cycles
    cycles = []
    if len(peaks) >= 2:
        # Last complete cycle
        cycles.append(slice(peaks[-2], peaks[-1]))
        if len(peaks) >= 3:
            # Second-to-last cycle
            cycles.append(slice(peaks[-3], peaks[-2]))
    else:
        cycles.append(slice(0, len(signal)))
    
    return cycles, cycle_period


def extract_last_cycle(time: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract last complete cardiac cycle from signal"""
    cycles, _ = find_cycles(time, signal)
    
    if not cycles:
        return time, signal
    
    last_cycle = cycles[0]
    return time[last_cycle], signal[last_cycle]


def calculate_hr_from_cycles(cycles: List[slice], time: np.ndarray) -> float:
    """Calculate heart rate from cycle periods"""
    if len(cycles) < 2 or len(cycles) > 2:
        # Use all available intervals
        cycle_times = [time[c.stop] - time[c.start] for c in cycles]
    else:
        # Use last two cycles
        cycle_times = [time[cycles[-1].stop] - time[cycles[-1].start]]
    
    if not cycle_times:
        return np.nan
    
    avg_cycle_time = np.mean(cycle_times)
    hr = 60.0 / avg_cycle_time if avg_cycle_time > 0 else np.nan
    return hr


def calculate_pressure_stats(pressure_signal: np.ndarray) -> Tuple[float, float, float]:
    """Calculate systolic, diastolic, and mean arterial pressure"""
    # Detect cycles to find repeated patterns
    cycles, _ = find_cycles(np.arange(len(pressure_signal)), pressure_signal)
    
    if cycles:
        last_cycle = cycles[0]
        cycle_p = pressure_signal[last_cycle]
    else:
        cycle_p = pressure_signal
    
    sys_p = np.max(cycle_p)
    dia_p = np.min(cycle_p)
    # MAP = diastolic + 1/3 * pulse pressure
    map_p = dia_p + (sys_p - dia_p) / 3.0
    
    return sys_p, dia_p, map_p


def calculate_stroke_volume(time: np.ndarray, flow: np.ndarray) -> float:
    """
    Calculate stroke volume by integrating flow over one cardiac cycle.
    
    Flow data is typically in mL/s, integration gives mL per cycle.
    Returns: stroke volume in mL
    """
    cycles, _ = find_cycles(time, np.abs(flow))
    
    if not cycles:
        cycles = [slice(0, len(flow))]
    
    # Use last complete cycle
    last_cycle = cycles[0]
    cycle_time = time[last_cycle]
    cycle_flow = flow[last_cycle]
    
    # Integrate using trapezoidal rule
    sv = np.trapz(np.abs(cycle_flow), cycle_time)
    
    return sv


def calculate_convergence(signal: np.ndarray) -> float:
    """
    Calculate convergence metric: RMS difference between last two cycles.
    
    Returns: normalized RMS difference (%)
    """
    cycles, _ = find_cycles(np.arange(len(signal)), signal)
    
    if len(cycles) < 2:
        # Insufficient data
        return 0.0
    
    cycle1 = signal[cycles[-1]]
    cycle2 = signal[cycles[-2]] if len(cycles) > 1 else signal[cycles[-1]]
    
    # Interpolate to same length
    min_len = min(len(cycle1), len(cycle2))
    if min_len < 10:
        return np.nan
    
    cycle1_interp = np.interp(np.linspace(0, 1, min_len), 
                               np.linspace(0, 1, len(cycle1)), cycle1)
    cycle2_interp = np.interp(np.linspace(0, 1, min_len), 
                               np.linspace(0, 1, len(cycle2)), cycle2)
    
    rms_diff = np.sqrt(np.mean((cycle1_interp - cycle2_interp) ** 2))
    rms_signal = np.sqrt(np.mean(cycle2_interp ** 2))
    
    if rms_signal > 0:
        return (rms_diff / rms_signal) * 100.0
    else:
        return 0.0


# ============================================================================
# Main Analysis Class
# ============================================================================
class FirstBloodAnalyzer:
    """Analyze FirstBlood simulation results"""
    
    def __init__(self, arterial_dir: Path):
        """
        Initialize analyzer with arterial output directory
        
        Args:
            arterial_dir: Path to arterial/ results directory
        """
        self.arterial_dir = Path(arterial_dir)
        if not self.arterial_dir.exists():
            raise FileNotFoundError(f"Arterial directory not found: {arterial_dir}")
        
        self.aorta_data = None
        self.flow_data = None
        self.cow_flows = {}
    
    def discover_signals(self) -> Dict[str, str]:
        """
        Auto-discover key arterial signals.
        
        Returns: dictionary with keys 'aorta_pressure', 'aorta_flow', 'cow_branches'
        """
        print(f"\n  Discovering arterial signals in {self.arterial_dir.name}/...")
        
        # Find aorta pressure file
        aorta_file = self._find_aorta_pressure()
        print(f"  ✓ Aorta pressure: {aorta_file}")
        
        # Find aorta flow file (same as pressure file for FirstBlood)
        flow_file = aorta_file  # In FirstBlood, flow is in same file
        print(f"  ✓ Aorta flow: {flow_file}")
        
        # Discover CoW branches
        found_branches = self._discover_cow_branches()
        print(f"  ✓ Found {len(found_branches)} CoW branches")
        
        return {
            'aorta_pressure': aorta_file,
            'aorta_flow': flow_file,
            'cow_branches': found_branches
        }
    
    def _find_aorta_pressure(self) -> str:
        """Find ascending aorta pressure file (typically A1.txt)"""
        # Try known aorta IDs
        for aorta_candidate in [AORTA_ID, 'A1', 'aorta', 'Aorta']:
            aorta_file = self.arterial_dir / f'{aorta_candidate}.txt'
            if aorta_file.exists():
                return aorta_candidate
        
        # Fallback: list candidates
        print("\n  ⚠ Could not find aorta pressure file!")
        print("  Candidate files (ranked by likelihood):")
        candidates = []
        for f in sorted(self.arterial_dir.glob('[A]*.txt')):
            fname = f.name.replace('.txt', '')
            # Rank by file characteristics
            rank = 0
            if fname == 'A1':
                rank = 10
            elif fname.startswith('A') and len(fname) <= 4:
                rank = 5
            candidates.append((rank, fname))
        
        candidates.sort(reverse=True)
        for rank, fname in candidates[:10]:
            print(f"    - {fname}")
        
        raise FileNotFoundError("Could not identify aorta pressure file")
    
    def _discover_cow_branches(self) -> Dict[str, str]:
        """Find available Circle of Willis branch files"""
        found = {}
        for file_id, (side, branch_name) in COW_VESSELS.items():
            file_path = self.arterial_dir / f'{file_id}.txt'
            if file_path.exists():
                found[file_id] = branch_name
        
        return found
    
    def load_aorta_data(self, aorta_id: str = AORTA_ID) -> bool:
        """Load aorta pressure and flow data"""
        file_path = self.arterial_dir / f'{aorta_id}.txt'
        if not file_path.exists():
            print(f"  ✗ Aorta file not found: {file_path}")
            return False
        
        data = load_timeseries(file_path)
        if data is None or data.shape[1] < 7:
            print(f"  ✗ Invalid aorta data format")
            return False
        
        self.aorta_data = data
        return True
    
    def calculate_cardiac_metrics(self) -> Optional[CardiacMetrics]:
        """Calculate cardiac hemodynamic metrics from aorta data"""
        if self.aorta_data is None:
            return None
        
        # Expected column layout from FirstBlood:
        # [time, P_in, P_out, v_avg, v_max, Q_in, Q_out, ..., d, ...]
        time = self.aorta_data[:, 0]
        p_in = self.aorta_data[:, 1]
        p_out = self.aorta_data[:, 2]
        q_in = self.aorta_data[:, 5]
        
        # Average inlet/outlet pressures and convert to gauge mmHg
        pressure_pa = (p_in + p_out) / 2.0
        pressure_gauge = pressure_to_mmhg(pressure_pa)
        
        # Calculate HR and cycle period
        cycles, cycle_period = find_cycles(time, pressure_gauge)
        heart_rate = calculate_hr_from_cycles(cycles, time)
        
        # Pressure stats (from last cycle)
        sys_p, dia_p, map_p = calculate_pressure_stats(pressure_gauge)
        
        # Stroke volume from flow
        # Convert flow from m³/s to mL/s
        flow_ml_s = q_in * 1e6
        stroke_volume = calculate_stroke_volume(time, flow_ml_s)
        
        # Cardiac output
        cardiac_output = (stroke_volume * heart_rate) / 1000.0 if not np.isnan(heart_rate) else np.nan
        
        # Convergence (periodicity)
        convergence = calculate_convergence(pressure_gauge)
        
        return CardiacMetrics(
            heart_rate=heart_rate,
            cycle_period=cycle_period,
            sys_pressure=sys_p,
            dia_pressure=dia_p,
            map_pressure=map_p,
            stroke_volume=stroke_volume,
            cardiac_output=cardiac_output,
            convergence_rms=convergence
        )
    
    def calculate_cerebral_flow(self) -> Optional[CerebralFlow]:
        """Calculate cerebral flow distribution in Circle of Willis"""
        # Load all CoW branch flows
        flows_by_branch = {}
        total_left = 0.0
        total_right = 0.0
        total_flow = 0.0
        
        for file_id, (side, branch_name) in COW_VESSELS.items():
            file_path = self.arterial_dir / f'{file_id}.txt'
            if not file_path.exists():
                continue
            
            data = load_timeseries(file_path)
            if data is None or data.shape[1] < 6:
                continue
            
            # Extract flow and integrate over last cycle
            time = data[:, 0]
            flow_in = data[:, 5]  # Column 5 is inlet flow
            flow_ml_s = flow_in * 1e6
            
            # Mean flow over last cycle
            cycles, _ = find_cycles(time, flow_ml_s)
            if cycles:
                last_cycle = cycles[0]
                cycle_flow = flow_ml_s[last_cycle]
                mean_flow = np.mean(np.abs(cycle_flow))
            else:
                mean_flow = np.mean(np.abs(flow_ml_s[-1000:]))
            
            flows_by_branch[branch_name] = mean_flow
            total_flow += mean_flow
            
            if side == 'R':
                total_right += mean_flow
            elif side == 'L':
                total_left += mean_flow
        
        if total_flow == 0:
            total_flow = 1.0  # Avoid division by zero
        
        lr_ratio = total_right / total_left if total_left > 0 else np.nan
        
        return CerebralFlow(
            total_flow=total_flow,
            left_flow=total_left,
            right_flow=total_right,
            left_right_ratio=lr_ratio,
            branches=flows_by_branch
        )
    
    def analyze(self) -> Tuple[CardiacMetrics, CerebralFlow]:
        """Run full analysis pipeline"""
        # Discover signals
        signals = self.discover_signals()
        
        # Load aorta
        if not self.load_aorta_data(signals['aorta_pressure']):
            raise RuntimeError("Failed to load aorta data")
        
        # Calculate metrics
        cardiac = self.calculate_cardiac_metrics()
        cerebral = self.calculate_cerebral_flow()
        
        if cardiac is None or cerebral is None:
            raise RuntimeError("Failed to calculate metrics")
        
        return cardiac, cerebral


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Compare FirstBlood simulation runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m validation.compare_runs --patient patient_025 --baseline Abel_ref2
  python -m validation.compare_runs \\
    --patient_dir ~/first_blood/projects/simple_run/results/patient_025 \\
    --baseline_dir ~/first_blood/projects/simple_run/results/Abel_ref2
        """
    )
    
    parser.add_argument('--patient', type=str, default='patient_025',
                       help='Patient model name (default: patient_025)')
    parser.add_argument('--baseline', type=str, default='Abel_ref2',
                       help='Baseline model name (default: Abel_ref2)')
    parser.add_argument('--patient_dir', type=Path,
                       help='Full path to patient results directory')
    parser.add_argument('--baseline_dir', type=Path,
                       help='Full path to baseline results directory')
    parser.add_argument('--simple_run_root', type=Path, 
                       default=Path.home() / 'final' / 'first_blood' / 'projects' / 'simple_run' / 'results',
                       help='Root directory for simple_run results')
    
    args = parser.parse_args()
    
    # Resolve paths
    if args.patient_dir:
        patient_dir = Path(args.patient_dir)
    else:
        patient_dir = args.simple_run_root / args.patient
    
    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
    else:
        baseline_dir = args.simple_run_root / args.baseline
    
    # Verify paths
    patient_arterial = patient_dir / 'arterial'
    baseline_arterial = baseline_dir / 'arterial'
    
    if not patient_arterial.exists():
        print(f"✗ Patient arterial directory not found: {patient_arterial}")
        sys.exit(1)
    
    if not baseline_arterial.exists():
        print(f"✗ Baseline arterial directory not found: {baseline_arterial}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("FirstBlood Validation: Run Comparison")
    print("="*70)
    
    try:
        # Analyze baseline
        print(f"\n[1/2] Analyzing baseline: {args.baseline}")
        baseline_analyzer = FirstBloodAnalyzer(baseline_arterial)
        baseline_cardiac, baseline_cerebral = baseline_analyzer.analyze()
        
        # Analyze patient
        print(f"\n[2/2] Analyzing patient: {args.patient}")
        patient_analyzer = FirstBloodAnalyzer(patient_arterial)
        patient_cardiac, patient_cerebral = patient_analyzer.analyze()
        
        # Create comparison
        comparison = ComparisonResult(
            timestamp=datetime.now().isoformat(),
            baseline_name=args.baseline,
            patient_name=args.patient,
            baseline_metrics=baseline_cardiac,
            patient_metrics=patient_cardiac,
            baseline_cerebral=baseline_cerebral,
            patient_cerebral=patient_cerebral,
        )
        
        # Generate outputs in validation directory
        validation_dir = Path(__file__).parent
        
        # Write summary text
        summary_file = validation_dir / f'{args.patient}_vs_{args.baseline}_summary.txt'
        write_summary_text(comparison, summary_file)
        
        # Write JSON
        json_file = validation_dir / f'{args.patient}_vs_{args.baseline}.json'
        write_json(comparison, json_file)
        
        print("\n" + "="*70)
        print("✓ Validation complete!")
        print(f"  Summary: {summary_file}")
        print(f"  JSON:    {json_file}")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def write_summary_text(comparison: ComparisonResult, output_file: Path):
    """Write human-readable summary"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FirstBlood Validation Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Timestamp: {comparison.timestamp}\n")
        f.write(f"Baseline:  {comparison.baseline_name}\n")
        f.write(f"Patient:   {comparison.patient_name}\n\n")
        
        # Cardiac metrics
        f.write("-"*80 + "\n")
        f.write("CARDIAC HEMODYNAMICS\n")
        f.write("-"*80 + "\n\n")
        
        baseline = comparison.baseline_metrics
        patient = comparison.patient_metrics
        
        f.write(f"{'Metric':<30} {'Baseline':>15} {'Patient':>15} {'Diff (%)':>15}\n")
        f.write("-"*80 + "\n")
        
        metrics_to_compare = [
            ('Heart Rate (bpm)', baseline.heart_rate, patient.heart_rate),
            ('Cycle Period (s)', baseline.cycle_period, patient.cycle_period),
            ('Systolic Pressure (mmHg)', baseline.sys_pressure, patient.sys_pressure),
            ('Diastolic Pressure (mmHg)', baseline.dia_pressure, patient.dia_pressure),
            ('Mean Arterial Pressure (mmHg)', baseline.map_pressure, patient.map_pressure),
            ('Stroke Volume (mL)', baseline.stroke_volume, patient.stroke_volume),
            ('Cardiac Output (L/min)', baseline.cardiac_output, patient.cardiac_output),
            ('Convergence RMS (%)', baseline.convergence_rms, patient.convergence_rms),
        ]
        
        for metric_name, baseline_val, patient_val in metrics_to_compare:
            if np.isnan(baseline_val) or np.isnan(patient_val):
                pct_diff = "N/A"
            else:
                if baseline_val != 0:
                    pct_diff = f"{((patient_val - baseline_val) / abs(baseline_val)) * 100:+.1f}%"
                else:
                    pct_diff = "N/A"
            
            f.write(f"{metric_name:<30} {baseline_val:>15.2f} {patient_val:>15.2f} {str(pct_diff):>15}\n")
        
        # Cerebral flow
        f.write("\n" + "-"*80 + "\n")
        f.write("CEREBRAL FLOW DISTRIBUTION (Circle of Willis)\n")
        f.write("-"*80 + "\n\n")
        
        baseline_ce = comparison.baseline_cerebral
        patient_ce = comparison.patient_cerebral
        
        f.write(f"{'Metric':<30} {'Baseline':>15} {'Patient':>15} {'Diff (%)':>15}\n")
        f.write("-"*80 + "\n")
        
        flow_metrics = [
            ('Total Cerebral Flow (mL/min)', baseline_ce.total_flow, patient_ce.total_flow),
            ('Left Flow (mL/min)', baseline_ce.left_flow, patient_ce.left_flow),
            ('Right Flow (mL/min)', baseline_ce.right_flow, patient_ce.right_flow),
            ('Left/Right Ratio', baseline_ce.left_right_ratio, patient_ce.left_right_ratio),
        ]
        
        for metric_name, baseline_val, patient_val in flow_metrics:
            if np.isnan(baseline_val) or np.isnan(patient_val):
                pct_diff = "N/A"
            else:
                if baseline_val != 0:
                    pct_diff = f"{((patient_val - baseline_val) / abs(baseline_val)) * 100:+.1f}%"
                else:
                    pct_diff = "N/A"
            
            f.write(f"{metric_name:<30} {baseline_val:>15.2f} {patient_val:>15.2f} {str(pct_diff):>15}\n")
        
        # Per-branch flows
        f.write("\nPer-Branch Flows (mL/min):\n")
        f.write(f"{'Branch':<20} {'Baseline':>15} {'Patient':>15}\n")
        f.write("-"*50 + "\n")
        
        all_branches = set(baseline_ce.branches.keys()) | set(patient_ce.branches.keys())
        # Filter out None values and sort
        all_branches = sorted([b for b in all_branches if b is not None])
        for branch in all_branches:
            b_flow = baseline_ce.branches.get(branch, 0.0)
            p_flow = patient_ce.branches.get(branch, 0.0)
            f.write(f"{branch:<20} {b_flow:>15.2f} {p_flow:>15.2f}\n")
        
        f.write("\n" + "="*80 + "\n")


def write_json(comparison: ComparisonResult, output_file: Path):
    """Write JSON output"""
    data = comparison.to_dict()
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()
