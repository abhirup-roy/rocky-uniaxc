import json
import os
import pathlib
from copy import deepcopy
from dataclasses import asdict
from typing import Callable, Optional, Protocol, runtime_checkable
import numpy as np
import medeq
from . import Scheduler
from ...pyrocky import Settings


class Constraints:
    def __init__(self, rules: list[Callable]):
        self.rules = rules

    def is_violated(self, params: dict) -> bool:
        for rule in self.rules:
            valid = rule(params)

            if not np.all(valid):
                return True
        return False


class ConstrainedDVASampler(medeq.DVASampler):
    def __init__(
        self, d, constraints: Optional[Constraints] = None, seed: Optional[int] = None
    ):
        super().__init__(d, seed=seed)
        self.constraints = constraints

    def cost(self, x, med: medeq.MED):
        base_cost = super().cost(x, med)
        x_reshaped = x.reshape(-1, self.d)

        mins = med.parameters["min"].to_numpy()
        maxs = med.parameters["max"].to_numpy()
        x_real = mins + (maxs - mins) * x_reshaped

        param_names = med.parameters.index.tolist()
        params_dict = {name: x_real[:, i] for i, name in enumerate(param_names)}

        if self.constraints is not None and self.constraints.is_violated(params_dict):
            return 1e9

        return base_cost


@runtime_checkable
class MEDSamplerLike(Protocol):
    def sample(self, x, med): ...


class RockyMED:
    def __init__(
        self,
        scheduler: Optional[Scheduler],
        sim_settings: Settings,
        med_config: dict[str, list[float | str]],
        sampler: Optional[MEDSamplerLike] = None,
        pyrocky_script_path: Optional[str] = None,
        seed: int = 42,
    ):
        if sampler is not None and not isinstance(sampler, MEDSamplerLike):
            raise ValueError("sampler must have a `sample` method")
        if not isinstance(sim_settings, Settings):
            raise ValueError("sim_settings must be an instance of Settings")

        self._validate_med_config(med_config)

        self.scheduler = scheduler
        self.sampler = sampler or medeq.DVASampler
        self.seed = int(seed)
        self.med_config = med_config
        self.sim_settings = sim_settings
        self.pyrocky_script_path = (
            pyrocky_script_path or pathlib.Path(__file__).parent / "_med_wrapper.py"
        )
        self.med: medeq.MED | None = None

    @staticmethod
    def _validate_med_config(med_config: dict) -> None:
        required_keys = {"variables", "minima", "maxima"}
        if not isinstance(med_config, dict):
            raise ValueError("med_config must be a dictionary")
        if med_config.keys() != required_keys:
            raise ValueError(f"med_config must have exactly keys: {required_keys}")
        if not all(isinstance(x, str) for x in med_config["variables"]):
            raise ValueError("variables must be a list of strings")
        if not all(isinstance(x, (int, float)) for x in med_config["minima"]):
            raise ValueError("minima must be a list of numbers")
        if not all(isinstance(x, (int, float)) for x in med_config["maxima"]):
            raise ValueError("maxima must be a list of numbers")

    def sample(self, x, med: medeq.MED):
        return self.sampler.sample(x, med)  # type: ignore

    def __call__(self, x, med: medeq.MED):
        return self.sample(x, med)

    def _serialize_full_config(self) -> str:
        config_data = deepcopy(self.med_config)
        config_data["settings"] = asdict(self.sim_settings)  # type: ignore

        return json.dumps(config_data)

    def run(self, n: int):

        config_str = self._serialize_full_config()
        os.environ["ROCKY_MED_CONFIG"] = config_str

        self.med = medeq.MED(
            self.pyrocky_script_path,
            scheduler=self.scheduler,
            sampler=self.sampler,
            seed=self.seed,
        )

        self.med.sample(n)
        self.med.evaluate()

        del os.environ["ROCKY_MED_CONFIG"]
