#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["ductile-loads[all]", "pyyaml"]
# ///
"""Loads processing v2 — OEM_loads_v2.yaml → ANSYS .inp files.

Design Practice: DP-TRS-LOADS-001, rev 1
Run: uv run loads_processing_v2.py
"""

import json
import yaml
from pathlib import Path
from ductile_loads import LoadSet

# --- Step 1: Read OEM delivery (YAML) and verify format ---
with open("OEM_loads_v2.yaml") as f:
    raw = yaml.safe_load(f)

print(f"Delivery: {raw['name']}, version {raw['version']}")
print(f"Units: forces={raw['units']['forces']}, moments={raw['units']['moments']}")
print(f"Load cases: {len(raw['load_cases'])}")

# --- Step 1b: Verify & fix interface point names ---
# OEM uses lug_left/lug_right; FEM expects lug_port/lug_starboard
NAME_MAP = {"lug_left": "lug_port", "lug_right": "lug_starboard"}

for lc in raw["load_cases"]:
    for pt in lc["point_loads"]:
        if pt["name"] in NAME_MAP:
            old = pt["name"]
            pt["name"] = NAME_MAP[old]
            print(f"  Renamed '{old}' → '{pt['name']}' in {lc['name']}")

# --- Step 2: Apply OEM Fx correction (×1.04) ---
for lc in raw["load_cases"]:
    for pt in lc["point_loads"]:
        pt["force_moment"]["fx"] *= 1.04

print("Applied Fx correction factor 1.04 to all interface points.")

# --- Step 3: Coordinate system — no transformation needed (DP §3) ---
print("Coordinate system: Engine CS = FEM CS. No transformation required.")

# --- Save corrected data as JSON for ductile-loads ---
json_path = Path("OEM_loads_v2_corrected.json")
with open(json_path, "w") as f:
    json.dump(raw, f, indent=2)

# --- Load into ductile-loads and process ---
loadset = LoadSet.read_json(str(json_path))

# --- Step 4: Unit conversion (DP §4) ---
# klbs/klbs.in → N/Nm
loadset_SI = loadset.convert_to("N")
print(f"Units converted: klbs → N, klbs.in → Nm")

# --- Step 5: Downselect via envelope (DP §5) ---
envelope = loadset_SI.envelope()
print(f"Envelope downselection: {len(loadset_SI.load_cases)} → {len(envelope.load_cases)} load cases")

# Rename "Limit_X" → "X" for clean filenames (DP §6.1)
for lc in envelope.load_cases:
    lc.name = (lc.name or "unnamed").split("_")[-1]

# --- Step 6: Write .inp files (DP §6) ---
envelope.to_ansys(folder_path="limit_loads", name_stem="limit_load", exclude=["bearing"])
print("ANSYS .inp files written to limit_loads/")

# --- Step 7a: Summary envelope table (DP §7.1) ---
md = envelope.envelope_to_markdown(output="envelope.md")
print("\n" + md)

envelope.get_point_extremes(output="envelope_extremes.json")
print("Envelope extremes written to envelope_extremes.json")

# --- Step 7b: Exceedance comparison against v1 (DP §7.2) ---
prev = LoadSet.read_json("previous_run/OEM_all_loads_v1.json")
prev_SI = prev.convert_to("N")
prev_envelope = prev_SI.envelope()

comparison = envelope.compare_to(prev_envelope)
print(f"\nExceedance check: new exceeds old = {comparison.new_exceeds_old()}")
comparison.generate_comparison_report(
    output_dir="comparison_report",
    report_name="v1_vs_v2",
)
print("Comparison report written to comparison_report/")
