# FirstBlood Data Generation Audit Report

**File:** `data_generation.py`  
**Date:** 2026-02-07  
**Status:** ⚠️ MULTIPLE CRITICAL ISSUES FOUND

---

## Executive Summary

The script **correctly implements the high-level pipeline** but has **8 critical safety gaps** that could cause:
- Silent failures (modified diameters that are anatomically invalid)
- Solver crashes (missing FB_IDs, invalid dimensions)
- Data quality issues (inconsistent unit conversions, missing validation)

---

## FOUND ISSUES

### 🔴 CRITICAL (Can cause crashes or data corruption)

#### **Issue #1: NO RANGE VALIDATION ON INPUT DIMENSIONS**
**Location:** Lines 560-580 (patient_geom build loop)  
**Problem:**
- No checks for negative/zero lengths or diameters
- No checks for anatomically implausible values (e.g., 0.01mm or 100mm)
- Cerebral vessels typically: length 5-50mm, diameter 0.5-5mm

**Example of failure:**
```python
# If JSON has corrupt data:
"radius_mm": -0.5  # Invalid!
# Converts to: diameter_m = (2 * -0.5) / 1000 = -0.001 ✗
# No error raised - silently injects negative diameter
```

**Fix needed:**
```python
# After line 577, before appending to candidates:
if length_m <= 0 or diameter_m <= 0:
    print(f"  ✗ SKIP {canon} ({side}): invalid dimensions L={length_m*1000}mm D={diameter_m*1000}mm")
    continue
if length_m > 0.1:  # > 100mm is implausible for CoW
    print(f"  ⚠️  WARNING {canon} ({side}): unusually long {length_m*1000}mm")
if diameter_m < 0.0001 or diameter_m > 0.01:  # Outside 0.1-10mm
    print(f"  ⚠️  WARNING {canon} ({side}): unusual diameter {diameter_m*1000}mm")
```

---

#### **Issue #2: SILENT FAILURE - FB_IDs NOT VALIDATED AGAINST TEMPLATE**
**Location:** Lines 608-630 (apply_geometry calls)  
**Problem:**
- `apply_geometry()` returns `None` if FB_ID not found in template
- No error tracking - user never knows injection failed
- Example: If template missing "A12", patient ICA geometry never injected

**Example of failure:**
```python
# Line 628: apply_geometry(df, "A12", 0.035, 0.0015)
# If "A12" not in arterial.csv:
#   apply_geometry() returns None (line 370-372)
#   No error, modification not tracked
#   Patient dataset silently uses template ICA geometry!
```

**Fix needed:**
```python
# Line 604: Validate all FB_IDs exist before modifications
fb_ids_in_template = set(df['ID'].dropna())
required_ids = set()
for id_list in fb_map.values():
    required_ids.update(id_list)
missing_ids = required_ids - fb_ids_in_template
if missing_ids:
    print(f"✗ ERROR: Template missing segments: {missing_ids}")
    print(f"  Cannot inject patient geometry - aborting")
    sys.exit(1)

# Later, track failed injections (lines 629, 652):
if mod is None:
    print(f"✗ INJECTION FAILED for {fb_id} - segment not found!")
    raise RuntimeError(f"Could not inject {fb_id}")
```

---

#### **Issue #3: BA SPLIT VALIDATION MISSING**
**Location:** Lines 608-619 (BA split)  
**Problem:**
- BA split into A59+A56 using template ratios
- No check that both FB_IDs exist
- No check that split ratio is reasonable (~1:1 to 1:3)

**Example of failure:**
```python
# Code does:
ba_ids = ["A59", "A56"]  # From fb_map
split_lens = split_length_by_template(df, ba_ids, 20.0)  # 20mm patient BA

# If template has:
#   A59: 5mm
#   A56: 5mm
# Result: split_lens = [10.0, 10.0]  ✓ OK

# But what if template has:
#   A59: 50mm (abnormal)
#   A56: 1mm  (tiny)
# Result: split_lens = [19.61mm, 0.39mm]  ✗ Unrealistic!
```

**Fix needed:**
```python
def split_length_by_template(df, ids, total_length_m):
    """Split with validation."""
    template_lengths = []
    for fb_id in ids:
        idxs = df.index[df["ID"] == fb_id].tolist()
        if not idxs:
            raise ValueError(f"FB_ID {fb_id} not found in template")
        tl = float(df.loc[idxs[0], "length[SI]"])
        template_lengths.append(tl)
    
    s = float(np.sum(template_lengths))
    split = [total_length_m * (tl / s) for tl in template_lengths]
    
    # Validate split is anatomically reasonable
    for i, (fb_id, length) in enumerate(zip(ids, split)):
        if length < 0.001 or length > 0.1:  # Outside 1-100mm
            fraction = length / total_length_m
            if fraction < 0.05 or fraction > 0.95:
                print(f"  ⚠️  WARNING: BA split gives {fb_id}={length*1000:.2f}mm ({fraction*100:.1f}%)")
    
    return split
```

---

#### **Issue #4: WALL THICKNESS RATIO NOT VALIDATED**
**Location:** Line 352  
**Problem:**
- Code assumes 10% diameter/thickness ratio uniformly: `thickness_m = 0.1 * diameter_m`
- NO verification that Abel_ref2 template actually uses this ratio
- If template uses 5% or 20%, patient model will have incorrect wall mechanics

**Example of failure:**
```python
# Template A12 has:
#   diameter: 4mm, thickness: 0.4mm (10% ratio)
#   Patient A12 gets: diameter 3mm, thickness 0.3mm (10% ratio)
#   BUT if template is actually 10% and solver expects this,
#   changing thickness breaks vessel mechanics!

# Worse: if different templates use different ratios,
# swapping templates breaks all models!
```

**Fix needed:**
```python
def get_template_thickness_ratio(df):
    """Extract the actual thickness/diameter ratio from template."""
    ratios = []
    for _, row in df.iterrows():
        d = float(row['start_diameter[SI]'])
        t = float(row['start_thickness[SI]'])
        if d > 0:
            ratios.append(t / d)
    
    if not ratios:
        return 0.1  # Fallback
    
    ratio = float(np.mean(ratios))
    print(f"  Template thickness/diameter ratio: {ratio:.3f}")
    
    # Warn if ratio is unusual
    if ratio < 0.05 or ratio > 0.25:
        print(f"  ⚠️  WARNING: Unusual template ratio {ratio}")
    
    return ratio

# In main(), after loading template arterial.csv:
thickness_ratio = get_template_thickness_ratio(df)

# Then in apply_geometry():
def apply_geometry(df, fb_id, length_m, diameter_m, thickness_ratio=0.1):
    ...
    thickness_m = thickness_ratio * diameter_m  # Use actual ratio
    ...
```

---

### 🟡 MAJOR (Can cause incorrect results)

#### **Issue #5: SIDE INFERENCE CONFLICT HANDLING**
**Location:** Lines 139-150  
**Problem:**
- When start_id and end_id have conflicting side hints (e.g., "R" vs "L")
- Code returns first hint instead of marking as ambiguous
- Silently masks data quality issues

**Example:**
```python
# If node path has:
#   start_id: 5 with hint "R"
#   end_id: 8 with hint "L"
# Code returns "R" (first hint) instead of None or raising error
# This could indicate:
#   - Parsing error in JSON
#   - Vessel crosses midline (pathological)
#   - Corrupted data
```

**Fix needed:**
```python
def side_from_endpoints(...):
    hints = []
    if start_id in node_side_hint:
        hints.append(node_side_hint[start_id])
    if end_id in node_side_hint:
        hints.append(node_side_hint[end_id])
    
    if hints:
        if all(h == hints[0] for h in hints):
            return hints[0]
        else:
            # Conflicting hints - this is suspicious!
            print(f"  ⚠️  CONFLICTING SIDES for segment {start_id}->{end_id}: {hints}")
            return None  # Don't silently pick first
```

---

#### **Issue #6: MISSING META.JSON OUTPUT**
**Location:** Lines 646-667  
**Problem:**
- Docstring promises: "Writes a meta.json with mapping decisions and conversions"
- Code never writes it - only writes `modifications_log.csv`
- Makes validation/reproducibility harder

**Fix needed:**
```python
# After line 665, before success message:
meta = {
    "patient_id": pid,
    "modality": modality,
    "template": args.template,
    "output_model": out_model_name,
    "vessel_count": len(modifications),
    "absent_vessels": len(absent_vessels),
    "patient_geometry": {
        str(k): {
            "length_mm": float(v["length_m"] * 1000),
            "diameter_mm": float(v["diameter_m"] * 1000),
        }
        for k, v in patient_geom.items()
    },
    "absent": [f"{s}_{c}" for s, c in absent_vessels],
    "modifications": modifications,
}

meta_path = out_dir / "meta.json"
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata: {meta_path}")
```

---

#### **Issue #7: ANATOMICAL VARIANT PARSING IS BRITTLE**
**Location:** Lines 315-331 (get_absent_vessels)  
**Problem:**
- Hardcoded section names: `["anterior", "posterior", "fetal"]`
- If variant JSON has different structure, silently ignores it
- Maps only specific vessel keys; what if JSON has different format?

**Example of failure:**
```python
# If new patient dataset has variants like:
{
    "vessel_presence": {  # Different key!
        "L-A1": false,
        ...
    }
}
# Current code won't find it (no "anterior" section)
# Silently processes patient as if all vessels present!
```

**Fix needed:**
```python
def get_absent_vessels(variants):
    """Parse variants robustly with error handling."""
    if not variants or not isinstance(variants, dict):
        print("  No variant data or invalid format")
        return []
    
    absent = []
    variant_to_key = {
        "L-A1": ("L", "A1"), "R-A1": ("R", "A1"),
        # ... existing mappings ...
    }
    
    # Check multiple possible section names
    checked_sections = []
    for section_name in ["anterior", "posterior", "fetal", "vessel_presence", "variants"]:
        if section_name not in variants:
            continue
        checked_sections.append(section_name)
        
        section = variants[section_name]
        if not isinstance(section, dict):
            print(f"  ⚠️  Section '{section_name}' is not dict - skipping")
            continue
        
        for var_key, present in section.items():
            if present is False and var_key in variant_to_key:
                absent.append(variant_to_key[var_key])
    
    if not checked_sections:
        print("  ⚠️  No recognized variant sections in file - assuming all vessels present")
    
    return absent
```

---

### 🟢 MINOR (Hygiene/maintainability)

#### **Issue #8: NORMALIZE_SEGMENT_NAME RETURNS WRONG_TYPE FOR SKIPPED VESSELS**
**Location:** Lines 235-255  
**Problem:**
- Returns `None` for "pca"/"aca" to skip them
- Later code: `if canon is None or canon not in CANONICAL: continue` (line 567)
- Works, but mixing `None` with string set membership is confusing

**Fix needed:**
```python
# Option 1: Use sentinel value
SKIP_VESSEL = object()

def normalize_segment_name(raw_name):
    ...
    if low in ("pca", "aca"):
        return SKIP_VESSEL  # More explicit
    ...

# Option 2: Return reason
def normalize_segment_name(raw_name):
    ...
    if low in ("pca", "aca"):
        return None, "combined_vessel"  # Tuple with reason
    ...
```

---

## VALIDATION CHECKLIST

Add these assertions to catch errors automatically:

```python
def validate_arterial_csv(df):
    """Validate arterial.csv for solver safety."""
    assert not df.empty, "arterial.csv is empty"
    assert "ID" in df.columns, "Missing ID column"
    assert "length[SI]" in df.columns, "Missing length[SI]"
    assert "start_diameter[SI]" in df.columns, "Missing diameter columns"
    
    # Check data types and ranges
    for idx, row in df.iterrows():
        length = float(row["length[SI]"])
        diam_start = float(row["start_diameter[SI]"])
        diam_end = float(row["end_diameter[SI]"])
        
        assert length > 0, f"Row {idx}: length <= 0"
        assert diam_start > 0, f"Row {idx}: start_diameter <= 0"
        assert diam_end > 0, f"Row {idx}: end_diameter <= 0"
        assert length < 0.5, f"Row {idx}: length > 500mm (implausible)"
        assert diam_start < 0.02, f"Row {idx}: diameter > 20mm (implausible)"
        assert diam_end < 0.02, f"Row {idx}: diameter > 20mm (implausible)"
        
        # Check nodes exist and are consistent
        start_node = str(row["start_node"]).strip()
        end_node = str(row["end_node"]).strip()
        assert start_node, f"Row {idx}: missing start_node"
        assert end_node, f"Row {idx}: missing end_node"
        assert start_node != end_node, f"Row {idx}: start==end node"

def validate_modifications(modifications, patient_geom):
    """Validate that all injections were successful."""
    injected_vessels = {m["FirstBlood_ID"] for m in modifications if m["Action"] == "inject"}
    expected_vessels = set()
    for (side, canon), _ in patient_geom.items():
        if canon == "BA":
            expected_vessels.update(["A59", "A56"])
        else:
            key = (None, canon) if canon == "Acom" else (side, canon)
            # ... add FB_IDs ...
    
    failed = expected_vessels - injected_vessels
    assert not failed, f"Failed to inject: {failed}"

def validate_meta_json(meta):
    """Validate meta.json structure."""
    assert "patient_id" in meta, "Missing patient_id"
    assert "vessel_count" in meta, "Missing vessel_count"
    assert meta["vessel_count"] > 0, "No modifications"
    assert "patient_geometry" in meta, "Missing patient_geometry"
```

---

## RECOMMENDED CODE CHANGES

### Priority 1: CRITICAL SECURITY/CORRECTNESS
1. **Add dimension range validation** (Issue #1)
2. **Validate FB_IDs exist before injection** (Issue #2)  
3. **Validate BA split reasonableness** (Issue #3)

### Priority 2: DATA INTEGRITY
4. **Make thickness ratio data-driven** (Issue #4)
5. **Document side inference conflicts** (Issue #5)
6. **Write meta.json output** (Issue #6)

### Priority 3: ROBUSTNESS
7. **Improve variant parsing flexibility** (Issue #7)
8. **Make segment skipping explicit** (Issue #8)

---

## TESTING RECOMMENDATIONS

```bash
# Test 1: Bad input dimensions
# Create test JSON with negative radius -> should be caught

# Test 2: Missing FB_IDs
# Create test template with missing segments -> should fail with clear error

# Test 3: Variant file format variations
# Test with different section names -> should warn but not crash

# Test 4: Output validation
# After generation, run:
python3 validate_model.py --model patient_025
# Should check:
#   - All CSV files readable
#   - Solver can initialize (no segfault)
#   - meta.json matches actual modifications
```

---

## SUMMARY TABLE

| Issue | Severity | Impact | Fix Time |
|-------|----------|--------|----------|
| #1: No dimension validation | 🔴 CRITICAL | Anatomically invalid models | 30 min |
| #2: No FB_ID validation | 🔴 CRITICAL | Silent injection failures | 45 min |
| #3: BA split validation | 🔴 CRITICAL | Unrealistic splits | 20 min |
| #4: Thickness ratio | 🟡 MAJOR | Incorrect mechanics | 45 min |
| #5: Side conflict handling | 🟡 MAJOR | Masks data issues | 20 min |
| #6: Missing meta.json | 🟡 MAJOR | Poor reproducibility | 15 min |
| #7: Variant parsing | 🟡 MAJOR | Breaks with new data format | 30 min |
| #8: Segment skip clarity | 🟢 MINOR | Code maintainability | 10 min |

**Total estimated fix time: ~3.5 hours**
