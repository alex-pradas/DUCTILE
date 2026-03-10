#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["ductile-loads[all]"]
# ///

from pathlib import Path
from ductile_loads import LoadSet

json_path = Path(__file__).parent / "OEM_loads_v2.json"
ls = LoadSet.read_json(str(json_path))

# Rename interface points
rename_map = {"lug_left": "lug_port", "lug_right": "lug_starboard", "lug_fairlead": "lug_failsafe"}
for lc in ls.load_cases:
    for pl in lc.point_loads:
        if pl.name in rename_map:
            pl.name = rename_map[pl.name]

# Remove bearing interface (reacted as boundary condition in FEM)
for lc in ls.load_cases:
    lc.point_loads = [pl for pl in lc.point_loads if pl.name != "bearing"]

# Apply 1.04 factor to Fx
for lc in ls.load_cases:
    for pl in lc.point_loads:
        pl.force_moment.fx *= 1.04

# Convert to Newtons
ls = ls.convert_to("N")

print(f"Name: {ls.name}")
print(f"Version: {ls.version}")
print(f"Units: {ls.units.forces}, {ls.units.moments}")
print(f"Load cases: {len(ls.load_cases)}")
print()

# Compute envelope
envelope = ls.envelope()
print(f"Envelope load cases: {len(envelope.load_cases)}")
print()
envelope.print_envelope()

# Export extremes and envelope markdown
output_dir = Path(__file__).parent
envelope.get_point_extremes(output=str(output_dir / "envelope_extremes.json"))
envelope.envelope_to_markdown(output=str(output_dir / "envelope.md"))

# Export to ANSYS (exclude bearing — already removed, but also via exclude for safety)
envelope.to_ansys(folder_path=str(output_dir / "limit_loads"), name_stem="limit_load")

# Exceedance comparison against previous loads (v1)
prev_path = output_dir / "previous_run" / "OEM_all_loads_v1.json"
prev_ls = LoadSet.read_json(str(prev_path))
comparison = ls.compare_to(prev_ls)
print()
if comparison.new_exceeds_old():
    print("WARNING: v2 loads exceed v1 envelope in one or more components.")
else:
    print("v2 loads are within the v1 envelope.")
comparison.generate_comparison_report(output_dir=str(output_dir / "comparison_report"))
