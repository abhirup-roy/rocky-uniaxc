from pandas.core.computation.ops import Op
from typing import Callable, Optional
import numpy as np
import medeq
import coexist
from coexist.schedulers import Scheduler, SlurmScheduler, LocalScheduler


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
