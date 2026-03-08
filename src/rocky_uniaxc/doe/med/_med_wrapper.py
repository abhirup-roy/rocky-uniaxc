import os
import json
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

run_config = replace(sim_settings, **med_subset)
with UniaxialCompressionSimulation(run_config) as sim:
    res = sim.execute(return_computed_metrics=True)

response = res
