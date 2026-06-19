# Command-Line Interface (CLI)

Rocky-UniaxC provides a command-line interface to execute a single simulation case using a saved `settings.json` configuration.

## Running a Case

To execute a case, call the case runner module directly and pass the path to a `settings.json` file:

```bash
python -m rocky_uniaxc.case_runner path/to/settings.json
```

This runner is invoked by the SLURM scheduler scripts generated inside each case directory.

## Settings File Format

Below is an example of a `settings.json` configuration file containing the parameters required by the runner to construct and execute the simulation:

```json
{
    "project_dir": "/path/to/project/directory",
    "particle_box_len": 0.1,
    "t_fill": 1.0,
    "t_settle": 0.5,
    "t_compress": 2.0,
    "p_compress": 1000.0,
    "p_radius": 0.005,
    "p_density": 2500.0,
    "p_youngmod": 10000000.0,
    "p_poisson": 0.25,
    "fric_dyn_pp": 0.3,
    "fric_stat_pp": 0.4,
    "cor_pp": 0.5,
    "fric_dyn_pw": 0.4,
    "fric_stat_pw": 0.5,
    "cor_pw": 0.5,
    "normal_force_model": "linear_hysteresis",
    "tangential_force_model": "coulomb_limit",
    "adhesion_model": "none",
    "rolling_fric": 0.1,
    "rolling_model": "none",
    "neighbor_search": "BVH",
    "processor": "GPU",
    "mesh_dir": "/path/to/meshes",
    "shape_name": "sphere",
    "vert_ar": 1.0,
    "horiz_ar": 1.0,
    "n_corners": 30,
    "sq_degree": 2.0,
    "particle_path": null,
    "smoothness": null
}
```
