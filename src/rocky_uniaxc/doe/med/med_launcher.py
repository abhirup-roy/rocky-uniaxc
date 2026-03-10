import json
import os
import pathlib
import pickle
import re
import subprocess
import time
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
        scheduler: Scheduler,
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
        self._slurm_job_ids: list[str] = []

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

    def _serialize_full_config(self) -> str:
        class _Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, pathlib.Path):
                    return str(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        config_data = deepcopy(self.med_config)
        config_data["settings"] = asdict(self.sim_settings)  # type: ignore

        return json.dumps(config_data, cls=_Encoder)

    def _setup_med(self) -> None:
        config_str = self._serialize_full_config()

        # Write config to a temp file that persists for the campaign lifetime
        self._config_file = (
            pathlib.Path(self.sim_settings.project_dir) / "rocky_med_config.json"
        )
        self._config_file.parent.mkdir(parents=True, exist_ok=True)
        self._config_file.write_text(config_str)

        # Set both: path for SLURM jobs, content for local fallback
        os.environ["ROCKY_MED_CONFIG_PATH"] = str(self._config_file)
        os.environ["ROCKY_MED_CONFIG"] = config_str

        self.med = medeq.MED(
            self.pyrocky_script_path,
            scheduler=self.scheduler,
            sampler=self.sampler,
            seed=self.seed,
        )

    def sample(self, n: int) -> None:
        if self.med is None:
            self._setup_med()
        self.med.sample(n)
        self.save(self._checkpoint_path())

    def submit(self) -> None:
        if self.med is None:
            raise RuntimeError("No MED instance found. Call `sample()` first.")

        popen = subprocess.Popen

        def _tracking_popen(cmd, **kwargs):
            proc = popen(cmd, **kwargs)
            if isinstance(cmd, list) and cmd[0] == "sbatch":
                stdout, _ = proc.communicate()
                # sbatch prints: "Submitted batch job <ID>"
                for line in (stdout or "").splitlines():
                    match = re.search(r"Submitted batch job (\d+)", line)
                    if match:
                        self._slurm_job_ids.append(match.group(1))
            return proc

        subprocess.Popen = _tracking_popen
        try:
            self.med.evaluate()
        finally:
            subprocess.Popen = popen
        self.save(self._checkpoint_path())
        print(
            f"Submitted {len(self._slurm_job_ids)} job(s) for this campaign: "
            f"{self._slurm_job_ids}\n"
            f"Checkpointed to {self._checkpoint_path()}.\n"
            f"Call `RockyMED.load(...).collect()` once jobs finish."
        )

    def _get_running_job_ids(self) -> list[str]:
        if not self._slurm_job_ids:
            return []
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "--noheader",
                    "--format=%i",
                    "--jobs",
                    ",".join(self._slurm_job_ids),  # query specific IDs only
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.split()
        except subprocess.CalledProcessError:
            # squeue returns non-zero if ALL queried jobs have finished
            return []

    def jobs_finished(self) -> bool:
        if not self._slurm_job_ids:
            raise RuntimeError(
                "No job IDs recorded. Call `submit()` before checking job status."
            )
        return len(self._get_running_job_ids()) == 0

    def wait_for_jobs(self, poll_interval: int = 60) -> None:
        """Block until all THIS campaign's jobs finish."""
        while not self.jobs_finished():
            remaining = self._get_running_job_ids()
            print(
                f"{len(remaining)} job(s) still running: {remaining}. "
                f"Checking again in {poll_interval}s..."
            )
            time.sleep(poll_interval)
        print("All campaign jobs finished.")

    def collect(self) -> np.ndarray:
        if self.med is None:
            raise RuntimeError(
                "No MED instance found. Load a saved campaign with "
                "`RockyMED.load()` first."
            )
        still_running = self._get_running_job_ids()
        if still_running:
            raise RuntimeError(
                f"Cannot collect: {len(still_running)} of THIS campaign's "
                f"SLURM job(s) are still running: {still_running}.\n"
                f"Call `wait_for_jobs()` or check `jobs_finished()` first."
            )
        responses = self.med.evaluate()
        self._cleanup()
        return responses

    def run(self, n: int) -> np.ndarray:
        self.sample(n)
        return self.collect()

    def _checkpoint_path(self) -> pathlib.Path:
        return pathlib.Path(self.sim_settings.project_dir) / "rocky_med_checkpoint.pkl"

    def _cleanup(self) -> None:
        os.environ.pop("ROCKY_MED_CONFIG", None)
        os.environ.pop("ROCKY_MED_CONFIG_PATH", None)
        # Only delete config file after collect() confirms all jobs finished
        if hasattr(self, "_config_file") and self._config_file.exists():
            self._config_file.unlink()

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "RockyMED":
        path = pathlib.Path(path)
        with open(path, "rb") as f:
            instance = pickle.load(f)

        if not isinstance(instance, cls):
            raise TypeError(
                f"Expected a RockyMED instance, got {type(instance).__name__}"
            )

        if instance.med is not None:
            config_str = instance._serialize_full_config()
            config_file = (
                pathlib.Path(instance.sim_settings.project_dir)
                / "rocky_med_config.json"
            )
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(config_str)
            instance._config_file = config_file
            os.environ["ROCKY_MED_CONFIG_PATH"] = str(config_file)
            os.environ["ROCKY_MED_CONFIG"] = config_str

        return instance

    def sample_and_submit(self, n: int) -> None:
        self.sample(n)
        self.submit()
