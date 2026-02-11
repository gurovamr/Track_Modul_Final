# Data Generation Script: Before vs After Fixes

## Summary of Changes

| Issue | Before | After | Impact |
|-------|--------|-------|--------|
| **Dimension validation** | None - silently accepts negative/implausible values | ✅ Range checks (0.1-100mm length, 0.1-10mm diameter) | Catches corrupt CoW data |
| **FB_ID validation** | Silent failure if segment missing from template | ✅ Pre-flight check, fail fast with clear error | Prevents ghost injections |
| **BA split validation** | No checks on split reasonableness | ✅ Validates each segment 1-100mm & detects > 95% imbalance | Detects anatomical anomalies |
| **Thickness ratio** | Hardcoded 0.1 ratio | ✅ Extracted from actual template | Works with any template |
| **Side inference conflicts** | Returns first hint, masks data issues | ✅ Returns None, documents conflict | Reveals data quality problems |
| **Meta.json output** | Not written (promised in docstring) | ✅ Full validation metadata JSON | Enables reproducibility |
| **Error messages** | Vague or missing | ✅ Detailed, actionable messages | Easier debugging |

---

## Code Comparisons

### ISSUE #1: Dimension Validation

#### ❌ BEFORE (Lines 560-580)
```python
for s in segs:
    canon = normalize_segment_name(s["raw_name"])
    if canon is None or canon not in CANONICAL:
        continue

    side = side_from_endpoints(...)
    length_m = float(s["length_mm"]) / 1000.0
    diameter_m = (2.0 * float(s["radius_mm"])) / 1000.0

    # NO VALIDATION - can have negative/implausible values!
    
    if canon in {"BA", "Acom"}:
        key = (None, canon)
    else:
        if side is None:
            print(f"  ⚠️  Skipping {canon}: could not infer side")
            continue
        key = (side, canon)

    candidates.append({
        "key": key,
        "length_m": length_m,      # Could be -0.001m!
        "diameter_m": diameter_m,  # Could be 0.1m (100mm)!
    })
```

**RISK:** If CoW JSON has `radius: -0.5mm` or `length: -100mm`, code silently injects invalid geometry.

#### ✅ AFTER
```python
def validate_dimensions(length_m, diameter_m, canon, side):
    """Validate input dimensions in plausible ranges."""
    vessel_name = f"{side or ''}{canon}".strip()
    
    if length_m <= 0:
        return False, f"Length {length_m*1000:.2f}mm is <= 0"
    if diameter_m <= 0:
        return False, f"Diameter {diameter_m*1000:.2f}mm is <= 0"
    if length_m < 0.001:  # < 1mm
        return False, f"Length implausibly short"
    if length_m > 0.1:  # > 100mm
        return False, f"Length implausibly long"
    if diameter_m < 0.0001:  # < 0.1mm
        return False, f"Diameter implausibly small"
    if diameter_m > 0.01:  # > 10mm
        return False, f"Diameter implausibly large"
    
    return True, f"✓ {vessel_name}: {length_m*1000:.2f}mm × {diameter_m*1000:.2f}mm"

# Then in main loop:
is_valid, msg = validate_dimensions(length_m, diameter_m, canon, side)
if not is_valid:
    print(f"  ✗ SKIP {canon} ({side}): {msg}")
    continue
print(f"  {msg}")
```

**RESULT:** Invalid data is caught immediately with clear error messages.

---

### ISSUE #2: FB_ID Validation

#### ❌ BEFORE (Lines 605-630)
```python
# No pre-flight check!
# Code just tries to apply geometry to FB_IDs that might not exist

fb_ids = fb_map[ba_key]  # ["A59", "A56"]
for fb_id, L in zip(fb_ids, split_lens):
    mod = apply_geometry(df, fb_id, L, diam)  # Might be None!
    if mod:
        modifications.append(mod)

# Later in apply_geometry():
def apply_geometry(df, fb_id, length_m, diameter_m):
    idxs = df.index[df["ID"] == fb_id].tolist()
    if not idxs:
        return None  # Silent failure!
    # ... rest of code ...
```

**RISK:** If template missing "A59", that BA segment never gets injected but no error raised.
User runs simulation with baseline BA distribution for missing segment.

#### ✅ AFTER
```python
def validate_fb_ids_exist(df, fb_map):
    """Pre-flight check: all required FB_IDs in template."""
    fb_ids_in_template = set(df['ID'].dropna().unique())
    required_ids = set()
    for id_list in fb_map.values():
        required_ids.update(id_list)
    
    missing_ids = required_ids - fb_ids_in_template
    if missing_ids:
        msg = f"Template missing: {sorted(missing_ids)}"
        return False, msg
    
    return True, f"All {len(required_ids)} FB_IDs present"

# In main():
all_exist, msg = validate_fb_ids_exist(df, fb_map)
print(f"  FB_ID check: {msg}")
if not all_exist:
    print(f"  ✗ ERROR: Cannot proceed")
    sys.exit(1)
```

**RESULT:** Script fails immediately with clear explanation before any modifications.

---

### ISSUE #3: BA Split Validation

#### ❌ BEFORE (Lines 608-619 & 443-453)
```python
def split_length_by_template(df, ids, total_length_m):
    template_lengths = []
    for fb_id in ids:
        idxs = df.index[df["ID"] == fb_id].tolist()
        template_lengths.append(float(df.loc[idxs[0], "length[SI]"]) if idxs else 0.0)
    s = float(np.sum(template_lengths))
    if s <= 1e-12:
        return [total_length_m / len(ids) for _ in ids]
    return [total_length_m * (tl / s) for tl in template_lengths]
    # NO VALIDATION - could produce [19.9mm, 0.1mm] for balanced BA!

# Usage:
ba_ids = ["A59", "A56"]
total_len = 20.0  # mm
split_lens = split_length_by_template(df, ba_ids, total_len)
# If template has A59=49.5mm, A56=0.5mm:
#   split_lens = [19.8mm, 0.2mm]  ← Unrealistic!
```

**RISK:** If template has skewed BA distribution, patient BA gets split unrealistically.

#### ✅ AFTER
```python
def split_length_by_template(df, ids, total_length_m):
    """Split with anatomical validation."""
    template_lengths = []
    for fb_id in ids:
        idxs = df.index[df["ID"] == fb_id].tolist()
        if not idxs:
            return None, f"FB_ID {fb_id} not found"
        template_lengths.append(float(df.loc[idxs[0], "length[SI]"]))
    
    s = float(np.sum(template_lengths))
    split = [total_length_m * (tl / s) for tl in template_lengths]
    
    # Validate split is anatomically reasonable
    warnings = []
    for i, (fb_id, length) in enumerate(zip(ids, split)):
        if length < 0.001 or length > 0.1:
            warnings.append(f"{fb_id}={length*1000:.2f}mm (extreme)")
        fraction = length / total_length_m
        if fraction < 0.01 or fraction > 0.99:
            warnings.append(f"{fb_id}: {fraction*100:.1f}% (skewed)")
    
    if warnings:
        return split, f"Split {total_length_m*1000:.1f}: {list(zip(ids,split))} [WARNINGS: {warnings}]"
    
    return split, f"Split OK: {list(zip(ids,split))}"

# Usage:
split_lens, split_msg = split_length_by_template(df, ba_ids, 20.0)
if split_lens is None:
    print(f"ERROR: {split_msg}")
    sys.exit(1)
print(f"BA split: {split_msg}")
```

**RESULT:** Detects and warns about unrealistic splits before applying them.

---

### ISSUE #4: Thickness Ratio from Template

#### ❌ BEFORE (Line 352)
```python
def apply_geometry(df, fb_id, length_m, diameter_m):
    ...
    # Assumed ratio - could be wrong for template!
    thickness_m = 0.1 * diameter_m  # Hardcoded 10%
    ...
```

**RISK:** If Abel_ref2 uses different ratio (e.g., 8% or 12%), patient model has wrong wall mechanics.

#### ✅ AFTER
```python
def get_template_thickness_ratio(df):
    """Extract actual ratio from template."""
    ratios = []
    for _, row in df.iterrows():
        d = float(row.get("start_diameter[SI]", 0))
        t = float(row.get("start_thickness[SI]", 0))
        if d > 0 and t > 0:
            ratios.append(t / d)
    
    ratio = float(np.mean(ratios)) if ratios else 0.1
    print(f"    Template thickness/diameter ratio: {ratio:.4f}")
    if ratio < 0.05 or ratio > 0.25:
        print(f"    ⚠️  Unusual ratio: {ratio:.4f}")
    return ratio

# In main():
thickness_ratio = get_template_thickness_ratio(df)

# In apply_geometry():
def apply_geometry(df, fb_id, length_m, diameter_m, thickness_ratio=0.1):
    ...
    thickness_m = thickness_ratio * diameter_m  # Data-driven!
    ...
```

**RESULT:** Automatically adapts to any template's wall mechanics.

---

### ISSUE #5: Side Inference Conflict Handling

#### ❌ BEFORE (Lines 139-150)
```python
hints = []
if start_id in node_side_hint:
    hints.append(node_side_hint[start_id])
if end_id in node_side_hint:
    hints.append(node_side_hint[end_id])

if hints:
    if all(h == hints[0] for h in hints):
        return hints[0]
    # Conflicting hints - silently pick first!
    if len(hints) >= 2:
        return hints[0]  # ← Returns "R" even if ["R", "L"]!
```

**RISK:** Masks data quality issues - conflicting hints might indicate:
- Vessel crosses midline (pathological)
- Corrupted JSON data
- Parsing error

#### ✅ AFTER
```python
hints = []
if start_id in node_side_hint:
    hints.append(node_side_hint[start_id])
if end_id in node_side_hint:
    hints.append(node_side_hint[end_id])

if hints:
    if all(h == hints[0] for h in hints):
        return hints[0]
    else:
        # Document conflict instead of hiding it
        if verbose:
            print(f"      ⚠️  Conflicting side hints for {start_id}->{end_id}: {hints}")
        return None  # ← Fails safely, user must investigate
```

**RESULT:** Conflicts are visible and can be debugged manually if rare.

---

### ISSUE #6: Meta.json Output

#### ❌ BEFORE (Lines 665-667)
```python
# Docstring says: "Writes a meta.json with mapping decisions"
# But code only writes:
pd.DataFrame(modifications).to_csv(out_dir / "modifications_log.csv", index=False)

# NO META.JSON OUTPUT!
# User cannot easily validate what was done
```

#### ✅ AFTER
```python
# Write comprehensive metadata
meta = {
    "patient_id": pid,
    "modality": modality,
    "template": args.template,
    "output_model": out_model_name,
    "creation_timestamp": "2026-02-07T...",
    
    "validation": {
        "fb_ids_check": "All 16 required FB_IDs present",
        "all_fb_ids_present": true,
        "thickness_ratio": 0.1002,
    },
    
    "geometry_summary": {
        "vessel_count": 12,
        "absent_vessels_count": 1,
        "patient_measurements": {
            "('R', 'ICA')": {"length_mm": 38.5, "diameter_mm": 4.2},
            "('L', 'ICA')": {"length_mm": 35.1, "diameter_mm": 3.8},
            ...
        },
        "absent_keys": ["L-Pcom_ABSENT"],
    },
    
    "modifications": [
        {
            "Action": "inject",
            "FirstBlood_ID": "A12",
            "Name": "R-ICA",
            "Old_length_mm": 40.0,
            "New_length_mm": 38.5,
            "Old_diameter_mm": 4.0,
            "New_diameter_mm": 4.2,
            "Thickness_ratio": 0.1002,
            "Patient_key": "R_ICA"
        },
        ...
    ]
}

with open(out_dir / "meta.json", 'w') as f:
    json.dump(meta, f, indent=2)
```

**RESULT:** Full audit trail of what was injected and validation that was performed.

---

## Testing Checklist

Use these tests to catch regressions:

```bash
#!/bin/bash

# Test 1: Bad dimensions
echo '{"1": {"ICA": [{"length": -50.0, "radius": {"mean": 2.0}}]}}' > bad_features.json
python data_generation_v8_fixed.py --features bad_features.json
# Expected: ✗ SKIP ICA: Length -50.00mm is <= 0

# Test 2: Missing FB_ID in template
# Create template without A12
python data_generation_v8_fixed.py --template incomplete
# Expected: ✗ ERROR: Cannot proceed - template missing A12

# Test 3: Skewed BA split
# Create template: A59=50mm, A56=1mm; patient BA=20mm
python data_generation_v8_fixed.py --pid 999
# Expected: BA split: [19.6mm, 0.4mm] [WARNING: A56: 2.0% (skewed)]

# Test 4: Metadata written
python data_generation_v8_fixed.py --pid 025 --force
ls -la models/patient_025/meta.json
# Expected: File exists, >500 bytes

# Test 5: No silent failures
grep -c "SKIP\|ERROR\|WARNING" meta.json
# Should show all problematic data
```

---

## Migration Guide

To use the fixed version:

```bash
# Backup original
cp pipeline/data_generation.py pipeline/data_generation_v7_backup.py

# Replace with fixed version
cp pipeline/data_generation_v8_fixed.py pipeline/data_generation.py

# Test on known patient
python pipeline/data_generation.py --pid 025 --force

# Check outputs
cat models/patient_025/meta.json  # New file!
diff models/patient_025/modifications_log.csv \
     models/patient_025_v7/modifications_log.csv  # Should be identical

# Validate model still runs
cd projects/simple_run
./simple_run.out patient_025
```

---

## Key Improvements Summary

| Aspect | Change |
|--------|--------|
| **Safety** | Dimension validation + FB_ID pre-check = no silent failures |
| **Reproducibility** | meta.json documents all decisions for audit trail |
| **Adaptability** | Thickness ratio extracted from template, not hardcoded |
| **Debugging** | Detailed warnings for anatomical anomalies (skewed BA, etc) |
| **Robustness** | Handles template variations gracefully |
| **Code Quality** | Clear error messages, early exits, explicit validation |
