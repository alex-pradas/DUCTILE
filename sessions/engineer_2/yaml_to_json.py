#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["pyyaml"]
# ///

import json
import yaml
from pathlib import Path

yaml_path = Path(__file__).parent / "OEM_loads_v2.yaml"
json_path = yaml_path.with_suffix(".json")

with open(yaml_path) as f:
    data = yaml.safe_load(f)

with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Converted {yaml_path.name} -> {json_path.name}")
