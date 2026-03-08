import hashlib
import os
import json
import pathlib
import medeq
from dataclasses import asdict, fields, replace
from ...pyrocky import Settings, UniaxialCompressionSimulation

config_str = os.environ.get("ROCKY_MED_CONFIG")
if not config_str:
    raise RuntimeError(
        "ROCKY_MED_CONFIG environment variable is missing. "
        "Did you run this through the RockyMED class?"
    )

config = json.loads(config_str)
sim_settings = Settings.from_dict(config.pop("settings"))

med_params = medeq.create_parameters(
    config["variables"], config["minima"], config["maxima"]
)
med_subset = med_params["value"].to_dict()

# Give each evaluation own project directory so runs never overwrite each other.
_param_hash = hashlib.sha1(
    json.dumps({k: round(v, 10) for k, v in sorted(med_subset.items())}).encode()
).hexdigest()[:12]
_base_project_dir = pathlib.Path(sim_settings.project_dir)
unique_project_dir = (
    _base_project_dir.parent / f"{_base_project_dir.name}_{_param_hash}"
)
unique_project_dir.mkdir(parents=True, exist_ok=True)

run_config = replace(sim_settings, project_dir=unique_project_dir, **med_subset)
with UniaxialCompressionSimulation(run_config) as sim:
    res = sim.execute(return_computed_metrics=True)

response = res
