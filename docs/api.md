# API Reference

This section documents the Python API of **Rocky-UniaxC**.

## Package-Level Settings

::: rocky_uniaxc.set_rocky_exe_path
::: rocky_uniaxc.set_headless_mode
::: rocky_uniaxc.set_backend

## Design of Experiments (DOE)

### Full Sweeps
::: rocky_uniaxc.doe.sweep.launch_sweep
::: rocky_uniaxc.doe.sweep.iter_params

### OFAT Sweeps
::: rocky_uniaxc.doe.ofat.launch_ofat
::: rocky_uniaxc.doe.ofat.iter_ofat

### DOE Utilities
::: rocky_uniaxc.doe._doe_utils.ShapeConfig
::: rocky_uniaxc.doe._doe_utils.SimParams
::: rocky_uniaxc.doe._doe_utils.case_directory
::: rocky_uniaxc.doe._doe_utils.render_pyrocky_script
::: rocky_uniaxc.doe._doe_utils.script_context_from_params
::: rocky_uniaxc.doe._doe_utils.get_unique_box_lens
::: rocky_uniaxc.doe._doe_utils.prepare_case

## Pyrocky Simulations
::: rocky_uniaxc.pyrocky.uniax.Settings
::: rocky_uniaxc.pyrocky.uniax.UniaxialCompressionSimulation

### Pyrocky Helpers
::: rocky_uniaxc.pyrocky.helpers.find_rocky_exe
::: rocky_uniaxc.pyrocky.helpers.pyrocky_run

## Particle Shapes
::: rocky_uniaxc.particles_shapes.Shape
::: rocky_uniaxc.particles_shapes.Sphere
::: rocky_uniaxc.particles_shapes.Polyhedron
::: rocky_uniaxc.particles_shapes.SpheroCylinder
::: rocky_uniaxc.particles_shapes.CustomPolyhedron

## Mesh Generation
::: rocky_uniaxc.compr_meshgen.create_meshes

## Post-Processing & Analysis
::: rocky_uniaxc.sweep_analysis.load_data
::: rocky_uniaxc.sweep_analysis.dump_results
::: rocky_uniaxc.sweep_analysis.find_faulty_runs
::: rocky_uniaxc.sweep_analysis.repeat_sweep

## External Formats (VTK/STL)
::: rocky_uniaxc.externals.generate_vtk
::: rocky_uniaxc.externals.export_particle_stl

## Scheduler & Utilities
::: rocky_uniaxc.utils.RockyScheduler
::: rocky_uniaxc.utils.cd
