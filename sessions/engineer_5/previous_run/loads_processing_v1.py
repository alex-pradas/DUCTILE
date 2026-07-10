#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["ductile-loads[all]"]
# ///
"""run this file using the command:  uv run loads_processing_v1.py
make sure you are in the same folder location.

"""

from ductile_loads import LoadSet

OEM_loadcase = LoadSet.read_json("OEM_all_loads_v1.json")
loadset_SI = OEM_loadcase.envelope().convert_to("N")

# Rename "Limit_X" → "X" for clean filenames (DP §6.1)
for lc in loadset_SI.load_cases:
    lc.name = (lc.name or "unnamed").split("_")[-1]

loadset_SI.to_ansys(folder_path="limit_loads", name_stem="limit_load", exclude=["bearing"])

print(loadset_SI.envelope_to_markdown(output="envelope.md"))

loadset_SI.get_point_extremes(output="envelope_extremes.json")
