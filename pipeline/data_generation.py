#!/usr/bin/env python3
"""
Patient-specific Circle of Willis model generator

Usage:
  python3 data_generation.py

Output:
  ~/first_blood/models/patient_<pid>/
    - arterial.csv (patient CoW geometry)
    - main.csv
    - p1.csv, ..., p47.csv (unchanged from Abel_ref2)
    - heart_kim_lit.csv (unchanged from Abel_ref2)
"""

import argparse
import json
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def get_repo_root() -> Path:
    pipeline_dir = Path(__file__).resolve().parent
    return pipeline_dir.parent


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _walk_json(obj, on_dict=None, on_list=None, path=()):
    if isinstance(obj, dict):
        if on_dict:
            on_dict(obj, path)
        for k, v in obj.items():
            _walk_json(v, on_dict=on_dict, on_list=on_list, path=path + (str(k),))
    elif isinstance(obj, list):
        if on_list:
            on_list(obj, path)
        for i, v in enumerate(obj):
            _walk_json(v, on_dict=on_dict, on_list=on_list, path=path + (str(i),))


def find_patient_files(data_root: Path, pid: str) -> Tuple[str, Path, Path, Optional[Path]]:
    """Find patient data files."""
    pid = str(pid).zfill(3)

    for modality in ["ct", "mr"]:
        feat_a = data_root / "cow_features" / f"topcow_{modality}_{pid}.json"
        nods_a = data_root / "cow_nodes" / f"topcow_{modality}_{pid}.json"
        var_a  = data_root / "cow_variants" / f"topcow_{modality}_{pid}.json"

        if feat_a.exists() and nods_a.exists():
            return modality, feat_a, nods_a, (var_a if var_a.exists() else None)
        
        nods_b = data_root / "cow_nodes" / f"nodes_{modality}_{pid}.json"
        if feat_a.exists() and nods_b.exists():
            return modality, feat_a, nods_b, (var_a if var_a.exists() else None)

    raise FileNotFoundError(f"Could not find patient files for pid={pid}")


def flatten_nodes(nodes_json: Any):
    """Pull out all the node positions and track which ones are on the right vs left."""
    id_to_xyz: Dict[int, np.ndarray] = {}
    node_side_hint: Dict[int, str] = {}
    r_x: List[float] = []
    l_x: List[float] = []

    def path_has_side(path) -> Optional[str]:
        # Walk through the JSON path looking for R- or L- labels
        for p in reversed(path):
            if not isinstance(p, str):
                continue
            if "R-" in p:
                return "R"
            if "L-" in p:
                return "L"
        return None

    def on_dict(d, path):
        if "id" in d and "coords" in d:
            try:
                nid = int(d["id"])
                coords = d["coords"]
                if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    xyz = np.array([float(coords[0]), float(coords[1]), float(coords[2])], dtype=float)
                    id_to_xyz[nid] = xyz

                    sh = path_has_side(path)
                    if sh in ("R", "L"):
                        # Track side hints, but only if they're consistent
                        if nid in node_side_hint and node_side_hint[nid] != sh:
                            node_side_hint.pop(nid, None)
                        else:
                            node_side_hint[nid] = sh
                            # Also collect all the x-coordinates for the right/left sides
                            if sh == "R":
                                r_x.append(float(xyz[0]))
                            else:
                                l_x.append(float(xyz[0]))
            except Exception:
                return

    _walk_json(nodes_json, on_dict=on_dict)
    return id_to_xyz, node_side_hint, r_x, l_x


def infer_right_is_positive_x(r_x: List[float], l_x: List[float]) -> bool:
    if len(r_x) > 0 and len(l_x) > 0:
        return float(np.mean(r_x)) > float(np.mean(l_x))
    return True


def side_from_endpoints(
    id_to_xyz: Dict[int, np.ndarray],
    node_side_hint: Dict[int, str],
    start_id: int,
    end_id: int,
    right_is_positive_x: bool,
    all_r_x: List[float] = None,
    all_l_x: List[float] = None
) -> Optional[str]:
    """Figure out if a vessel segment is on the right or left side.
    
    First the start/end nodes that have any explicit R/L labels are checked.
    If not,  their x-coordinates are compared to the midline between all known right and left nodes.
    """
    hints = []
    if start_id in node_side_hint:
        hints.append(node_side_hint[start_id])
    if end_id in node_side_hint:
        hints.append(node_side_hint[end_id])

    if hints:
        if all(h == hints[0] for h in hints):
            return hints[0]
       
        if len(hints) >= 2:
            return hints[0]

    # Get the x-coordinates of the start and end nodes if available
    xs = []
    if start_id in id_to_xyz:
        xs.append(float(id_to_xyz[start_id][0]))
    if end_id in id_to_xyz:
        xs.append(float(id_to_xyz[end_id][0]))
    
    if not xs:
        return None
    
    x_mean = float(np.mean(xs))
    
    # If x-coords for all right and left nodes exist, use the midline between them
    if all_r_x and all_l_x:
        midline = (np.mean(all_r_x) + np.mean(all_l_x)) / 2.0
        return "R" if x_mean > midline else "L"
    
    # Fallback: just check the sign relative to the origin
    if right_is_positive_x:
        return "R" if x_mean >= 0.0 else "L"
    return "R" if x_mean <= 0.0 else "L"


def parse_patient_features(features_json: Any) -> List[Dict[str, Any]]:
    """Go through the JSON and pull out all the vessel segment data."""
    segs: List[Dict[str, Any]] = []

    def on_dict(d, path):
        if "segment" not in d or "length" not in d or "radius" not in d:
            return
        seg = d.get("segment", {})
        rad = d.get("radius", {})
        if not isinstance(seg, dict) or not isinstance(rad, dict):
            return
        if "start" not in seg or "end" not in seg or "mean" not in rad:
            return

        try:
            start_id = int(seg["start"])
            end_id = int(seg["end"])
            length_mm = float(d["length"])
            radius_mm = float(rad["mean"])
        except Exception:
            return

        # vessel name from the JSON path 
        raw_name = None
        for p in reversed(path):
            if not str(p).isdigit():
                raw_name = str(p)
                break
        if raw_name is None:
            raw_name = "UNKNOWN"

        # Skip bifurcations and keep the actual vessels
        if "bifurcation" in raw_name.lower():
            return

        segs.append({
            "raw_name": raw_name.strip(),
            "start_id": start_id,
            "end_id": end_id,
            "length_mm": length_mm,
            "radius_mm": radius_mm,
        })

    _walk_json(features_json, on_dict=on_dict)
    return segs


CANONICAL = {"A1", "A2", "Acom", "Pcom", "MCA", "BA", "P1", "P2", "ICA", "C6", "C7"}

def normalize_segment_name(raw_name: str) -> str:
    """Mapping from input data to match standard vessel names."""
    name = str(raw_name).strip()
    if name in CANONICAL:
        return name
    low = name.lower().replace(" ", "")
    if low == "acom":
        return "Acom"
    if low == "pcom":
        return "Pcom"
    if low == "ba":
        return "BA"
    # C6 and C7 are  different sections of the internal carotid artery
    if low in ("c6", "c7"):
        return "ICA"
    #  PCA and ACA are grouped together with A2 and P2
    if low in ("pca", "aca"):
        return None
    return name


def build_firstblood_mapping() -> Dict[Tuple[Optional[str], str], List[str]]:
    return {
        ("R", "ICA"): ["A12"],
        ("L", "ICA"): ["A16"],
        ("R", "MCA"): ["A70"],
        ("L", "MCA"): ["A73"],
        ("R", "A1"): ["A68"],
        ("L", "A1"): ["A69"],
        ("R", "A2"): ["A76"],
        ("L", "A2"): ["A78"],
        ("R", "P1"): ["A60"],
        ("L", "P1"): ["A61"],
        ("R", "P2"): ["A64"],
        ("L", "P2"): ["A65"],
        ("R", "Pcom"): ["A62"],
        ("L", "Pcom"): ["A63"],
        (None, "Acom"): ["A77"],
        (None, "BA"): ["A59", "A56"],
    }


def get_absent_vessels(variants: Optional[Dict]) -> List[Tuple[Optional[str], str]]:
    """
    Extract which vessels are missing/abnormal from the variant file.
    
    The variant JSON has sections for anterior/posterior/fetal circulation,
    and each vessel is marked true (present) or false (absent).
    """
    if not variants:
        return []
    
    absent = []
    
    # Convert the variant keys to the correct format
    variant_to_key = {
        "L-A1": ("L", "A1"), "R-A1": ("R", "A1"),
        "L-A2": ("L", "A2"), "R-A2": ("R", "A2"),
        "Acom": (None, "Acom"),
        "L-Pcom": ("L", "Pcom"), "R-Pcom": ("R", "Pcom"),
        "L-P1": ("L", "P1"), "R-P1": ("R", "P1"),
        "L-P2": ("L", "P2"), "R-P2": ("R", "P2"),
        "L-MCA": ("L", "MCA"), "R-MCA": ("R", "MCA"),
    }
    
    for section in ["anterior", "posterior", "fetal"]:
        if section not in variants:
            continue
        for var_key, present in variants[section].items():
            if present is False and var_key in variant_to_key:
                absent.append(variant_to_key[var_key])
    
    return absent


def mark_vessel_absent(df: pd.DataFrame, fb_id: str) -> Optional[Dict[str, Any]]:
    """
    Mark anatomically absent vessels by assigning minimal diameter (0.1 mm) to effectively 
    block hemodynamic flow through non-existent structures.
    """
    idxs = df.index[df["ID"] == fb_id].tolist()
    if not idxs:
        return None
    i = idxs[0]
    
    name = str(df.loc[i, "name"])
    old_diam = float(df.loc[i, "start_diameter[SI]"])
    
    # Minimal diameter assignment to prevent flow in absent vessel
    absent_diameter = 0.0001
    absent_thickness = 0.00001
    
    df.loc[i, "start_diameter[SI]"] = absent_diameter
    df.loc[i, "end_diameter[SI]"] = absent_diameter
    df.loc[i, "start_thickness[SI]"] = absent_thickness
    df.loc[i, "end_thickness[SI]"] = absent_thickness
    
    return {
        "Action": "mark_absent",
        "FirstBlood_ID": fb_id,
        "Name": name,
        "Old_diameter_mm": old_diam * 1000.0,
        "New_diameter_mm": absent_diameter * 1000.0,
        "Reason": "Anatomical variant - vessel absent",
    }


def apply_geometry(df: pd.DataFrame, fb_id: str, length_m: float, diameter_m: float) -> Optional[Dict[str, Any]]:
    """Update a vessel's length, diameter, and wall thickness with patient data.
    
    The wall thickness is always 10% of the diameter (ratio is taken from Abel_ref2 ).
    """
    idxs = df.index[df["ID"] == fb_id].tolist()
    if not idxs:
        return None
    i = idxs[0]

    old_length = float(df.loc[i, "length[SI]"])
    old_d1 = float(df.loc[i, "start_diameter[SI]"])
    name = str(df.loc[i, "name"])

    # Wall thickness should be 10% of diameter
    thickness_m = 0.1 * diameter_m

    df.loc[i, "length[SI]"] = float(length_m)
    df.loc[i, "start_diameter[SI]"] = float(diameter_m)
    df.loc[i, "end_diameter[SI]"] = float(diameter_m)
    # the wall thickness is updated so it scales with the new diameter
    df.loc[i, "start_thickness[SI]"] = float(thickness_m)
    df.loc[i, "end_thickness[SI]"] = float(thickness_m)

    return {
        "Action": "inject",
        "FirstBlood_ID": fb_id,
        "Name": name,
        "Old_length_mm": old_length * 1000.0,
        "New_length_mm": length_m * 1000.0,
        "Old_diameter_mm": old_d1 * 1000.0,
        "New_diameter_mm": diameter_m * 1000.0,
    }


def split_length_by_template(df: pd.DataFrame, ids: List[str], total_length_m: float) -> List[float]:
    # Total length is split proportionally based on how long each segment is in the template
    template_lengths = []
    for fb_id in ids:
        idxs = df.index[df["ID"] == fb_id].tolist()
        template_lengths.append(float(df.loc[idxs[0], "length[SI]"]) if idxs else 0.0)
    s = float(np.sum(template_lengths))
    # If the template has no length data, it is  split equally
    if s <= 1e-12:
        return [total_length_m / len(ids) for _ in ids]
    # Otherwise scale each segment by its proportion of the total
    return [total_length_m * (tl / s) for tl in template_lengths]


def find_terminal_nodes(arterial_df: pd.DataFrame) -> List[str]:
    """
    Find all the peripheral terminal nodes (p1, p2, etc.) where vessels end in the model.
    """
    # Find nodes that are end points but never start points
    starts = set(arterial_df['start_node'].dropna())
    ends = set(arterial_df['end_node'].dropna())
    terminals = ends - starts
    # Filter to just the p1, p2, ... nodes 
    peripheral_pattern = re.compile(r'^p\d+$')
    peripherals = sorted([n for n in terminals if peripheral_pattern.match(str(n))])
    return peripherals


def generate_correct_main_csv(arterial_df: pd.DataFrame, output_path: Path, time_duration: float = 10.317):
    """
    Generate main.csv with the right connections between peripherals and arterial nodes.
    
    Each peripheral pX needs to connect to the exact node that feeds it in the arterial network.
    The key is mapping which node feeds each peripheral terminal.
    """
    peripherals = find_terminal_nodes(arterial_df)
    
    if not peripherals:
        raise ValueError("No peripheral nodes (p1, p2, ...) found in arterial.csv")
    
    print(f"  Found {len(peripherals)} peripheral terminal nodes")
    
    # Build a map of which arterial node feeds into each peripheral
    peripheral_to_node = {}
    for _, row in arterial_df.iterrows():
        if row['type'] != 'vis_f':
            continue
        end_node = str(row['end_node']).strip()
        if end_node.startswith('p') and end_node[1:].isdigit():
            start_node = str(row['start_node']).strip()
            peripheral_to_node[end_node] = start_node
            print(f"    {end_node} <- {start_node}")
    
    lines = [
        "run,forward",
        f"time,{time_duration}",
        "material,linear",
        "solver,maccormack",
        "",
        "type,name,main node,model node,main node,model node,...",
    ]
    
    # MOC arterial connections (pairing each peripheral to its model node)
    moc_connections = []
    for p in sorted(peripherals):
        num = p[1:]  
        main_node = f"N{num}p"
        moc_connections.extend([main_node, p])
    moc_connections.extend(["Heart", "H"])
    
    lines.append("moc,arterial," + ",".join(moc_connections))
    lines.append("")
    
    # Connection of each peripheral lumped network to the arterial node 
    for p in sorted(peripherals):
        num = p[1:]
        main_node = f"N{num}p"
        feeding_node = peripheral_to_node.get(p, 'n1')
        lines.append(f"lumped,{p},{main_node},{feeding_node}")
    
    lines.append("lumped,heart_kim_lit,Heart,aorta")
    lines.append("")
    
    for p in sorted(peripherals):
        num = p[1:]
        lines.append(f"node,N{num}p")
    lines.append("node,Heart")
    lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Generated main.csv:")
    print(f"    {len(peripherals)} peripherals with correct feeding connections")


def main():
    ap = argparse.ArgumentParser(
        description="Generate a patient-specific Circle of Willis model by modifying template geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""How it works:
  1. Start with the Abel_ref2 template 
  2. SwAsjust the CoW vessel lengths and diameters with patient data
  3. Keep peripherals & main.csv

Example:
  python3 data_generation.py
        """
    )
    ap.add_argument("--pid", default=None, help="Patient ID (e.g., 025)")
    ap.add_argument("--template", default="Abel_ref2", help="Template model")
    ap.add_argument("--force", action="store_true", help="Overwrite existing")
    args = ap.parse_args()

    pid = args.pid
    if pid is None or str(pid).strip() == "":
        pid = input("Enter patient id (e.g. 025): ").strip()
    pid = str(pid).zfill(3)

    out_model_name = f"patient_{pid}"

    repo_root = get_repo_root()
    data_root = repo_root / "data"
    template_dir = repo_root / "models" / args.template
    out_dir = repo_root / "models" / out_model_name

    if not template_dir.exists():
        raise FileNotFoundError(f"Template not found: {template_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    modality, feature_path, nodes_path, variant_path = find_patient_files(data_root, pid)


    features = load_json(feature_path)
    nodes = load_json(nodes_path)
    
    # Load anatomical variants
    variants = None
    absent_vessels = []
    if variant_path and variant_path.exists():
        variants = load_json(variant_path)
        absent_vessels = get_absent_vessels(variants)
        if absent_vessels:
            print(f"Anatomical variants loaded: {len(absent_vessels)} absent vessels")
            for av in absent_vessels:
                side, canon = av
                print(f"  - {side or ''}-{canon}: ABSENT")

    # Extract all the node positions and figure out the left/right orientation
    id_to_xyz, node_side_hint, r_x, l_x = flatten_nodes(nodes)
    right_is_pos_x = infer_right_is_positive_x(r_x, l_x)

    # vessel segments from the patient data
    segs = parse_patient_features(features)
    print(f"Parsed {len(segs)} vessel segments")

    # Mappin vessels to FirstBlood IDs
    fb_map = build_firstblood_mapping()
    candidates: List[Dict[str, Any]] = []

    # Filtering for required segments
    for s in segs:
        canon = normalize_segment_name(s["raw_name"])
        if canon is None or canon not in CANONICAL:
            continue

        # Determine which side the vessel is on
        side = side_from_endpoints(id_to_xyz, node_side_hint, s["start_id"], s["end_id"], right_is_pos_x, r_x, l_x)
        # Convert measurements from mm to meters
        length_m = float(s["length_mm"]) / 1000.0
        diameter_m = (2.0 * float(s["radius_mm"])) / 1000.0

        # Acom and BA don't have a side because they are midline structures
        if canon in {"BA", "Acom"}:
            key = (None, canon)
        else:
            if side is None:
                print(f"   Skipping {canon}: could not infer side (nodes {s['start_id']}->{s['end_id']})")
                continue
            key = (side, canon)

        candidates.append({
            "key": key,
            "canon": canon,
            "side": side,
            "length_m": length_m,
            "diameter_m": diameter_m,
        })

    # the best measurement for each vessel is kept
    patient_geom: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
    for c in candidates:
        key = c["key"]
        if key not in patient_geom or float(c["length_m"]) > float(patient_geom[key]["length_m"]):
            patient_geom[key] = c

    print(f"Kept {len(patient_geom)} unique CoW measurements")
    print("-" * 78)

    # Create output directory
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {out_dir}\nUse --force")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_arterial = template_dir / "arterial.csv"
    if not template_arterial.exists():
        raise FileNotFoundError(f"Missing arterial.csv in template")

    # Copy all the template files 
    print("\nStep 1: Copying template files...")
    copied = 0
    for src in template_dir.glob("*.csv"):
        shutil.copy(src, out_dir / src.name)
        copied += 1
    print(f"  Copied {copied} CSV files")

    # Load the arterial.csv file and insert patient data
    print("\nStep 2:  patient-specific CoW geometry...")
    arterial_path = out_dir / "arterial.csv"
    df = pd.read_csv(arterial_path)

    modifications: List[Dict[str, Any]] = []

    # Update the basilar artery first
    ba_key = (None, "BA")
    if ba_key in patient_geom:
        fb_ids = fb_map[ba_key]
        total_len = float(patient_geom[ba_key]["length_m"])
        diam = float(patient_geom[ba_key]["diameter_m"])
        # Split the total BA length across the segments proportionally
        split_lens = split_length_by_template(df, fb_ids, total_len)

        for fb_id, L in zip(fb_ids, split_lens):
            mod = apply_geometry(df, fb_id, L, diam)
            if mod:
                mod.update({"Patient_key": "BA"})
                modifications.append(mod)

    # Update the other major vessels
    for (side, canon), g in patient_geom.items():
        if canon == "BA":
            continue

        map_key = (None, canon) if canon == "Acom" else (side, canon)
        if map_key not in fb_map:
            continue

        for fb_id in fb_map[map_key]:
            mod = apply_geometry(df, fb_id, float(g["length_m"]), float(g["diameter_m"]))
            if mod:
                pk = canon if side is None else f"{side}_{canon}"
                mod.update({"Patient_key": pk})
                modifications.append(mod)

    # Handle any anatomical variants
    if absent_vessels:
        print(f"\nStep 2b: Marking {len(absent_vessels)} absent vessels...")
        for av_key in absent_vessels:
            if av_key in fb_map:
                for fb_id in fb_map[av_key]:
                    mod = mark_vessel_absent(df, fb_id)
                    if mod:
                        side, canon = av_key
                        pk = f"{side or ''}-{canon}_ABSENT"
                        mod.update({"Patient_key": pk})
                        modifications.append(mod)
                        print(f"    {fb_id} ({mod['Name']}): marked absent")

    # Save modified arterial.csv
    df.to_csv(arterial_path, index=False)
    print(f"  Modified {len(modifications)} CoW vessels total")

    print("\nStep 3: Keeping template main.csv...")

    # Save log
    pd.DataFrame(modifications).to_csv(out_dir / "modifications_log.csv", index=False)


    print("=" * 78)
    print(f"Output: {out_dir}")
    print(f"\nRun simulation:")
    print(f"  cd {repo_root}/projects/simple_run")
    print(f"  ./simple_run.out {out_model_name}")
    print(f"\nValidate results:")
    print(f"  python3 {repo_root}/pipeline/validation.py --model {out_model_name}")
    print("=" * 78)


if __name__ == "__main__":
    main()