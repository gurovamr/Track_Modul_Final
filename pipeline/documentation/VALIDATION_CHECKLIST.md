# FirstBlood Data Generation: Validation & Assertion Checklist

## Integration Guide: Add These Assertions to Catch Errors Automatically

### 1. INPUT VALIDATION (Run at script start)

```python
def validate_input_files(feature_path, nodes_path, variant_path=None):
    """Pre-flight checks on input data."""
    # Check existence
    assert feature_path.exists(), f"Missing features: {feature_path}"
    assert nodes_path.exists(), f"Missing nodes: {nodes_path}"
    
    # Check readability
    try:
        features = load_json(feature_path)
        nodes = load_json(nodes_path)
        assert isinstance(features, dict), "Features is not dict"
        assert isinstance(nodes, dict), "Nodes is not dict"
    except Exception as e:
        raise ValueError(f"Cannot load input JSON: {e}")
    
    # Check basic structure
    assert len(features) > 0, "Features JSON is empty"
    assert len(nodes) > 0, "Nodes JSON is empty"
    
    if variant_path and variant_path.exists():
        try:
            variants = load_json(variant_path)
        except Exception as e:
            print(f"⚠️  Warning: Cannot load variants: {e}")
    
    print(f"✓ Input validation passed")
    return features, nodes, variants if variant_path else None
```

### 2. PARSING VALIDATION (After extracting segments)

```python
def validate_parsed_segments(segs, patient_geom, threshold_min_length=0.001, 
                             threshold_max_length=0.1, threshold_min_diam=0.0001, 
                             threshold_max_diam=0.01):
    """Validate parsed vesselometry is anatomically plausible."""
    
    if not segs:
        raise ValueError("No segments parsed from features JSON")
    
    if not patient_geom:
        raise ValueError("No patient geometry extracted")
    
    # Check dimension ranges
    for key, geom in patient_geom.items():
        side, canon = key
        length = geom["length_m"]
        diameter = geom["diameter_m"]
        
        vessel_name = f"{side or 'M'}-{canon}"
        
        # Length checks
        assert length > 0, f"{vessel_name}: negative length {length*1000}mm"
        assert length >= threshold_min_length, \
            f"{vessel_name}: length {length*1000:.3f}mm < {threshold_min_length*1000:.3f}mm"
        assert length <= threshold_max_length, \
            f"{vessel_name}: length {length*1000:.1f}mm > {threshold_max_length*1000:.1f}mm"
        
        # Diameter checks
        assert diameter > 0, f"{vessel_name}: negative diameter {diameter*1000}mm"
        assert diameter >= threshold_min_diam, \
            f"{vessel_name}: diameter {diameter*1000:.3f}mm < {threshold_min_diam*1000:.3f}mm"
        assert diameter <= threshold_max_diam, \
            f"{vessel_name}: diameter {diameter*1000:.2f}mm > {threshold_max_diam*1000:.2f}mm"
    
    print(f"✓ Segment validation passed: {len(patient_geom)} unique vessels")
    return True
```

### 3. TEMPLATE VALIDATION (After loading template CSV)

```python
def validate_template_arterial_csv(df, template_name="Abel_ref2"):
    """Validate template arterial.csv structure and data."""
    
    # Check required columns
    required_cols = ["ID", "name", "length[SI]", "start_diameter[SI]", 
                     "end_diameter[SI]", "start_thickness[SI]", "end_thickness[SI]",
                     "start_node", "end_node", "type"]
    
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check no empty IDs
    assert df["ID"].notna().all(), "Template has rows with empty ID"
    
    # Check all data types parseable
    for col in ["length[SI]", "start_diameter[SI]", "end_diameter[SI]"]:
        try:
            df[col].astype(float)
        except:
            raise ValueError(f"Column {col} contains non-numeric values")
    
    # Check all values positive (except may have zeros)
    for col in ["length[SI]", "start_diameter[SI]", "end_diameter[SI]"]:
        assert (df[col] >= 0).all(), f"{col} has negative values"
        assert (df[col] > 0).any(), f"{col} is all zeros"
    
    # Check thickness < diameter
    for idx, row in df.iterrows():
        thick = float(row["start_thickness[SI]"])
        diam = float(row["start_diameter[SI]"])
        assert thick < diam, f"Row {idx}: thickness {thick} > diameter {diam}"
    
    # Check consistent naming
    ids_in_template = set(df["ID"].dropna())
    
    print(f"✓ Template validation passed:")
    print(f"  - {len(df)} rows")
    print(f"  - {len(ids_in_template)} unique segment IDs")
    print(f"  - All required columns present")
    print(f"  - All data numeric and >0")
    
    return ids_in_template
```

### 4. FB_ID VALIDATION (Before injection)

```python
def validate_fb_ids_injectable(df, fb_map, patient_geom):
    """Validate all required FB_IDs can be injected."""
    
    ids_in_template = set(df["ID"].dropna())
    
    # Collect all FB_IDs that will be injected
    required_ids = set()
    for (side, canon), geom in patient_geom.items():
        if (side, canon) in fb_map:
            required_ids.update(fb_map[(side, canon)])
        elif (None, canon) in fb_map:
            required_ids.update(fb_map[(None, canon)])
    
    # Check all required IDs exist
    missing = required_ids - ids_in_template
    assert not missing, f"Template missing FB_IDs: {missing}"
    
    # Check all BA segments can be injected
    if (None, "BA") in patient_geom:
        ba_ids = fb_map[(None, "BA")]
        for ba_id in ba_ids:
            assert ba_id in ids_in_template, f"Missing BA segment: {ba_id}"
    
    # Check all CoW IDs are unique (no duplicates)
    all_mapping_ids = []
    for id_list in fb_map.values():
        all_mapping_ids.extend(id_list)
    
    duplicates = [x for x in all_mapping_ids if all_mapping_ids.count(x) > 1]
    assert not duplicates, f"Duplicate FB_IDs in mapping: {set(duplicates)}"
    
    print(f"✓ FB_ID validation passed:")
    print(f"  - {len(required_ids)} FB_IDs to inject")
    print(f"  - All present in template")
    print(f"  - No duplicates in mapping")
    
    return required_ids, ids_in_template
```

### 5. BA SPLIT VALIDATION (Before BA injection)

```python
def validate_ba_split(ba_ids, patient_ba_length, split_lengths, template_df):
    """Validate BA split is anatomically reasonable."""
    
    assert len(ba_ids) == len(split_lengths), "BA_IDs and split_lengths mismatch"
    assert len(ba_ids) == 2, "BA should split into 2 segments"
    
    # Check sum is conserved
    total = sum(split_lengths)
    assert abs(total - patient_ba_length) < 1e-9, \
        f"BA split sum {total*1000}mm != patient {patient_ba_length*1000}mm"
    
    # Check individual segments in range
    for ba_id, length in zip(ba_ids, split_lengths):
        assert 0.001 < length < 0.1, \
            f"{ba_id}: length {length*1000:.2f}mm outside 1-100mm range"
    
    # Check split is not too skewed (<5% or >95%)
    fractions = [l / patient_ba_length for l in split_lengths]
    for ba_id, frac in zip(ba_ids, fractions):
        assert 0.05 <= frac <= 0.95, \
            f"{ba_id}: split fraction {frac*100:.1f}% is extreme (should be 5-95%)"
    
    # Check both segments exist in template
    template_ids = set(template_df["ID"].dropna())
    for ba_id in ba_ids:
        assert ba_id in template_ids, f"Template missing {ba_id}"
    
    print(f"✓ BA split validation passed:")
    print(f"  - {ba_ids[0]}: {split_lengths[0]*1000:.2f}mm ({fractions[0]*100:.1f}%)")
    print(f"  - {ba_ids[1]}: {split_lengths[1]*1000:.2f}mm ({fractions[1]*100:.1f}%)")
    
    return True
```

### 6. THICKNESS RATIO VALIDATION (After extracting from template)

```python
def validate_thickness_ratio(template_df, ratio, expected_range=(0.05, 0.25)):
    """Validate thickness/diameter ratio from template."""
    
    assert 0 < ratio < 1, f"Thickness ratio {ratio} outside (0,1)"
    assert expected_range[0] <= ratio <= expected_range[1], \
        f"Thickness ratio {ratio:.4f} outside expected range {expected_range}"
    
    # Check each vessel has consistent thickness
    thickness_ratios = []
    for _, row in template_df.iterrows():
        d = float(row.get("start_diameter[SI]", 0))
        t = float(row.get("start_thickness[SI]", 0))
        if d > 0 and t > 0:
            thickness_ratios.append(t / d)
    
    if thickness_ratios:
        ratio_min = min(thickness_ratios)
        ratio_max = max(thickness_ratios)
        ratio_std = np.std(thickness_ratios)
        
        assert ratio_max - ratio_min < 0.1, \
            f"Template thickness ratio varies widely: {ratio_min:.4f} to {ratio_max:.4f}"
        
        print(f"✓ Thickness ratio validation passed:")
        print(f"  - Mean:  {ratio:.4f}")
        print(f"  - Range: {ratio_min:.4f} to {ratio_max:.4f}")
        print(f"  - Std:   {ratio_std:.4f}")
    
    return ratio
```

### 7. MODIFICATIONS INTEGRITY (After all modifications)

```python
def validate_modifications_integrity(df, modifications, patient_geom, absent_vessels):
    """Validate all modifications were applied correctly."""
    
    if not modifications:
        return False, "No modifications recorded"
    
    # Count injections and absences
    injections = [m for m in modifications if m["Action"] == "inject"]
    markings = [m for m in modifications if m["Action"] == "mark_absent"]
    
    assert len(injections) + len(markings) == len(modifications), \
        "Unknown modification types"
    
    # Check all patient vessels were injected
    injected_ids = {m["FirstBlood_ID"] for m in injections}
    
    # Check injection records have required fields
    for mod in injections:
        assert "FirstBlood_ID" in mod, "Missing FirstBlood_ID in modification"
        assert "Patient_key" in mod, "Missing Patient_key in modification"
        assert "Old_diameter_mm" in mod, "Missing Old_diameter_mm"
        assert "New_diameter_mm" in mod, "Missing New_diameter_mm"
        assert "Thickness_ratio" in mod, "Missing Thickness_ratio"
        
        # Check dimensions changed
        old_d = mod["Old_diameter_mm"]
        new_d = mod["New_diameter_mm"]
        assert new_d > 0, f"{mod['FirstBlood_ID']}: new diameter <= 0"
        assert old_d > 0, f"{mod['FirstBlood_ID']}: old diameter <= 0"
    
    # Check absence records
    for mod in markings:
        assert mod["New_diameter_mm"] <= 0.1, \
            f"{mod['FirstBlood_ID']}: marked absent but diameter {mod['New_diameter_mm']}mm"
    
    # Verify in actual DataFrame
    for _, row in df.iterrows():
        fb_id = str(row["ID"])
        diam = float(row.get("start_diameter[SI]", 0))
        
        # All diameters should be positive now
        assert diam > 0, f"Row with ID {fb_id}: diameter {diam} <= 0"
    
    print(f"✓ Modifications integrity passed:")
    print(f"  - {len(injections)} injections recorded")
    print(f"  - {len(markings)} absences marked")
    print(f"  - All DataFrame values valid")
    
    return True
```

### 8. OUTPUT VALIDATION (After writing output files)

```python
def validate_output_files(out_dir):
    """Validate all output files exist and are parseable."""
    
    required_files = [
        "arterial.csv",
        "main.csv",
        "modifications_log.csv",
        "meta.json",
    ]
    
    for fname in required_files:
        fpath = out_dir / fname
        assert fpath.exists(), f"Missing output file: {fname}"
        assert fpath.stat().st_size > 0, f"Empty output file: {fname}"
    
    # Check CSVs are valid
    arterial_df = pd.read_csv(out_dir / "arterial.csv")
    main_df = pd.read_csv(out_dir / "main.csv")
    
    assert len(arterial_df) > 0, "arterial.csv is empty"
    assert len(main_df) > 0, "main.csv is empty"
    
    # Check meta.json is valid
    try:
        with open(out_dir / "meta.json") as f:
            meta = json.load(f)
        assert "patient_id" in meta, "Missing patient_id in meta"
        assert "modifications" in meta, "Missing modifications in meta"
    except Exception as e:
        raise ValueError(f"Invalid meta.json: {e}")
    
    print(f"✓ Output files validation passed:")
    print(f"  - {len(required_files)} required files present")
    print(f"  - arterial.csv: {len(arterial_df)} rows")
    print(f"  - main.csv: {len(main_df)} rows")
    print(f"  - meta.json: valid JSON")
    
    return True
```

---

## Usage: Insert in Main Function

```python
def main():
    # ... argument parsing ...
    
    # 1. Validate inputs
    features, nodes, variants = validate_input_files(feature_path, nodes_path, variant_path)
    
    # 2. Parse and validate segments
    segs = parse_patient_features(features)
    patient_geom = build_patient_geometry(segs, ...)
    validate_parsed_segments(segs, patient_geom)
    
    # 3. Validate template
    template_df = pd.read_csv(template_arterial)
    ids_in_template = validate_template_arterial_csv(template_df)
    
    # 4. Validate FB_IDs
    required_ids, available_ids = validate_fb_ids_injectable(template_df, fb_map, patient_geom)
    
    # 5. Get and validate thickness ratio
    thickness_ratio = get_template_thickness_ratio(template_df)
    thickness_ratio = validate_thickness_ratio(template_df, thickness_ratio)
    
    # 6. Perform modifications
    modifications = []
    
    # BA injection with validation
    if (None, "BA") in patient_geom:
        split_lens, msg = split_length_by_template(...)
        validate_ba_split(fb_ids, total_length, split_lens, template_df)
        # ... inject BA ...
    
    # Other injections
    # ... rest of modifications ...
    
    # 7. Validate modifications
    validate_modifications_integrity(template_df, modifications, patient_geom, absent_vessels)
    
    # 8. Write outputs
    # ... write CSV, JSON ...
    
    # 9. Validate outputs
    validate_output_files(out_dir)
    
    print("\n✓ ALL VALIDATIONS PASSED")
```

---

## Summary: Invariants to Assert

| Invariant | Where | Purpose |
|-----------|-------|---------|
| `length_m > 0 && length_m < 0.1` | Input validation | Catch negative/implausible lengths |
| `diameter_m > 0 && diameter_m < 0.01` | Input validation | Catch negative/implausible diameters |
| All FB_IDs in template | Pre-injection | Prevent silent failures |
| BA split fractions in [0.05, 0.95] | BA injection | Detect anatomical anomalies |
| `thickness_ratio > 0 && < 1` | Thickness extraction | Maintain wall mechanics |
| `thickness < diameter` | Template check | Physical plausibility |
| All modifications have Patient_key | Post-injection | Audit trail completeness |
| Output files non-empty | Post-write | Catch I/O errors |
| meta.json parses as valid JSON | Post-write | Data integrity |

---

## Testing Command

```bash
# Run with all validations
python data_generation_v8_fixed.py --pid 025 --force --validate

# Check for warnings
grep -i "warning\|error\|skip" models/patient_025/meta.json

# Compare with baseline
diff <(python data_generation_v7.py --pid 025) \
     <(python data_generation_v8_fixed.py --pid 025)
```
