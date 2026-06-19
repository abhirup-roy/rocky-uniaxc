# Examples

This page shows how to run simulations using the high-level API in Rocky-UniaxC.

## 1. Running a Parameter Sweep

To run a full-factorial parameter sweep, you define the parameter space in a JSON file and launch it using `launch_sweep`.

Create a `sweep_config.json` configuration:

```json
{
  "shape": {
    "name": "sphere"
  },
  "particle_properties": {
    "radius": [0.005, 0.006],
    "density": [2500],
    "poisson": [0.25],
    "youngmod": [1e7]
  },
  "inseractions": {
    "pp": {
      "fric_dyn": [0.3, 0.5],
      "fric_stat": [0.4],
      "fric_rolling": [0.1],
      "cor": [0.5]
    },
    "pw": {
      "fric_dyn": [0.4],
      "fric_stat": [0.5],
      "cor": [0.5]
    }
  },
  "experim_settings": {
    "box_len": [0.1],
    "p_compress": [1000]
  },
  "contact_model": {
    "normal": "linear_hysteresis",
    "tangential": "coulomb_limit",
    "rolling": "none",
    "adhesion": "none"
  }
}
```

Then, load a scheduler preset (e.g., BlueBear GPU cluster scheduler) and launch:

```python
from rocky_uniaxc import launch_sweep
from rocky_uniaxc.utils import RockyScheduler

# Define the SLURM scheduler settings
scheduler = RockyScheduler.bb_gpu(ngpus=1, run_days=1, account="my-slurm-account")

# Generate cases and submit jobs to SLURM queue
launch_sweep(
    sweep_name="my_first_sweep",
    scheduler=scheduler,
    json_path="sweep_config.json",
    autolaunch=True,
    target="GPU",
    backend="pyrocky"
)
```

## 2. Running an OFAT Experiment Block

One-Factor-at-a-Time (OFAT) designs vary a single parameter while holding others constant. Define the parameters you want to vary and the base levels in a JSON file:

```python
from rocky_uniaxc import launch_ofat
from rocky_uniaxc.utils import RockyScheduler

scheduler = RockyScheduler.bb_cpu(ncpus=4, run_days=1)

ofat_specs = {
    "parameters": ["fric_dyn_pp", "cor_pp"],
    "test_range": [(0.1, 0.8), (0.2, 0.9)],
    "hold_values": ["l", "m"],  # Strategy: hold at low or mid levels
}

launch_ofat(
    sweep_name="my_ofat_sweep",
    scheduler=scheduler,
    ofat_values=ofat_specs,
    n_points=5,
    json_path="base_config.json",
    autolaunch=True,
    target="CPU"
)
```

## 3. Post-Processing and Analyzing Sweep Results

After simulations complete, you can query database files, check for failures/lost particles, and summarize results.

```python
import rocky_uniaxc.sweep_analysis as analyze

# Load results from SQLite into a pandas DataFrame
df = analyze.load_data("my_first_sweep")
print(df.head())

# Find runs that had particle loss or weird Hausner ratios
analyze.find_faulty_runs("my_first_sweep", dump=True)

# Export combined results to CSV or Parquet
analyze.dump_results("my_first_sweep", filetype="parquet", minimal=True)
```
