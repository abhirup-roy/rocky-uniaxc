"""External file format handlers for Rocky simulation data.

Provides utilities for exporting particle geometries from Rocky as STL,
converting them to VTK unstructured grids with per-cell data (orientation,
velocity, etc.), and generating VTK files for visualisation in ParaView.
"""

from typing import Optional, Any
import os
import tempfile
from warnings import warn
import numpy as np
from stl import mesh
from evtk import vtk, hl
from .pyrocky import pyrocky_run


def _get_rotation_matrix(vec: np.ndarray) -> np.ndarray:
    """Compute a rotation matrix that aligns the z-axis with the given vector.

    Uses Rodrigues' rotation formula.

    Args:
        vec: A 3-element rotation vector (axis × angle).

    Returns:
        A 3×3 rotation matrix as a :class:`~numpy.ndarray`.
    """
    angle = np.linalg.norm(vec)

    axis = vec / angle
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def _load_particle_stl(file_path: str):
    """Load a particle STL file and return its vertices and face indices.

    Args:
        file_path: Path to the STL file.

    Returns:
        A tuple ``(vertices, faces)`` where ``vertices`` is an
        ``(N, 3)`` array and ``faces`` is an ``(M, 3)`` array of
        indices.  Returns ``(None, None)`` on failure.
    """
    try:
        p_mesh = mesh.Mesh.from_file(file_path)
        verts, inv_idxs = np.unique(
            p_mesh.vectors.reshape(-1, 3), axis=0, return_inverse=True
        )
        faces = inv_idxs.reshape(-1, 3)
        return verts, faces

    except Exception as e:
        print(f"Error loading STL file: {e}")
        return None, None


def export_particle_stl(project: Any, study: Any) -> str:
    """Export the first particle's geometry to an STL file.

    Args:
        project: Rocky project API object.
        study: Rocky study API object.

    Returns:
        Path to the exported STL file.
    """

    study = project.GetStudy()
    particle = study.GetParticleCollection()[0]
    tempdir = tempfile.mkdtemp(prefix="rocky_particle")
    stl_path = os.path.join(tempdir, "particle.stl")
    export_toolkit = study.GetExportToolkit()
    export_toolkit.ExportParticleToStl(
        stl_filename=stl_path, particle=particle, time_to_export=-1
    )

    return stl_path


def _vtk_gen(
    stl_path: str,
    positions: np.ndarray,
    orientations: np.ndarray,
    trans_vel_data: np.ndarray,
    rotat_vel_data: np.ndarray,
    residence_times: np.ndarray,
    output_vtk_path: str,
):
    """Convert STL particle data and per-particle attributes to VTK format.

    Instantiates the template mesh at each particle position with the
    correct rotation and writes an unstructured-grid VTK file with
    per-cell data (orientation, velocity, etc.).

    Args:
        stl_path: Path to the template particle STL file.
        positions: ``(N, 3)`` array of particle positions.
        orientations: ``(N, 3)`` array of rotation vectors.
        trans_vel_data: ``(N, 3)`` array of translational velocities.
        rotat_vel_data: ``(N, 3)`` array of rotational velocities.
        residence_times: ``(N,)`` array of residence times.
        output_vtk_path: Output path for the VTK file (without extension).
    """
    template_verts, template_faces = _load_particle_stl(stl_path)
    if template_verts is None or template_faces is None:
        print("Failed to load STL file.")
        return

    all_vertices_list = []
    all_connectivity_list = []
    cell_data_lists = {
        "particle_id": [],
        "orientation_x": [],
        "orientation_y": [],
        "orientation_z": [],
        "transl_vel_x": [],
        "transl_vel_y": [],
        "transl_vel_z": [],
        "rotat_vel_x": [],
        "rotat_vel_y": [],
        "rotat_vel_z": [],
    }

    n_verts_per_particle = template_verts.shape[0]
    n_faces_per_particle = template_faces.shape[0]
    current_vertex_offset = 0

    print(positions)

    n_particles = positions.shape[0]
    for n in range(n_particles):
        pos = positions[n]
        orient = orientations[n]
        vel_trans = trans_vel_data[n]
        vel_rotat = rotat_vel_data[n]

        r_mat = _get_rotation_matrix(orient)
        rotated_verts = template_verts @ r_mat.T

        translated_verts = rotated_verts + pos
        all_vertices_list.append(translated_verts)

        offset_faces = template_faces + current_vertex_offset
        all_connectivity_list.append(offset_faces)

        cell_data_lists["particle_id"].append(np.full(n_faces_per_particle, n + 1))
        cell_data_lists["orientation_x"].append(
            np.full(n_faces_per_particle, orient[0])
        )
        cell_data_lists["orientation_y"].append(
            np.full(n_faces_per_particle, orient[1])
        )
        cell_data_lists["orientation_z"].append(
            np.full(n_faces_per_particle, orient[2])
        )
        cell_data_lists["transl_vel_x"].append(
            np.full(n_faces_per_particle, vel_trans[0])
        )
        cell_data_lists["transl_vel_y"].append(
            np.full(n_faces_per_particle, vel_trans[1])
        )
        cell_data_lists["transl_vel_z"].append(
            np.full(n_faces_per_particle, vel_trans[2])
        )
        cell_data_lists["rotat_vel_x"].append(
            np.full(n_faces_per_particle, vel_rotat[0])
        )
        cell_data_lists["rotat_vel_y"].append(
            np.full(n_faces_per_particle, vel_rotat[1])
        )
        cell_data_lists["rotat_vel_z"].append(
            np.full(n_faces_per_particle, vel_rotat[2])
        )

        current_vertex_offset += n_verts_per_particle

    if not all_vertices_list:
        warn("No particles were processed. Check input data.")

    vertices = np.vstack(all_vertices_list)
    connectivity = np.concatenate(all_connectivity_list).flatten()

    cellData = {key: np.concatenate(value) for key, value in cell_data_lists.items()}
    n_total_faces = len(cellData["particle_id"])
    cell_types = np.full(n_total_faces, vtk.VtkTriangle.tid)

    vx, vy, vz = (
        cellData.pop("transl_vel_x"),
        cellData.pop("transl_vel_y"),
        cellData.pop("transl_vel_z"),
    )
    omegax, omegay, omegaz = (
        cellData.pop("rotat_vel_x"),
        cellData.pop("rotat_vel_y"),
        cellData.pop("rotat_vel_z"),
    )

    cellData["velocity"] = (vx, vy, vz)
    cellData["angular_velocity"] = (omegax, omegay, omegaz)

    x, y, z = vertices[:, 0].copy(), vertices[:, 1].copy(), vertices[:, 2].copy()
    offsets = np.arange(3, 3 * n_total_faces + 1, 3)

    print(f"Writing VTK file to {output_vtk_path}...")
    hl.unstructuredGridToVTK(
        output_vtk_path,
        x=x,
        y=y,
        z=z,
        connectivity=connectivity,
        offsets=offsets,
        cell_types=cell_types,
        cellData=cellData,
        pointData=None,
    )
    print("VTK file written successfully.")


@pyrocky_run()
def generate_vtk(rocky: Any, rocky_filepath: str, output_dir: str) -> None:
    """Generate VTK files for particle data at every saved time step.

    Reads particle positions, orientations, translational and rotational
    velocities from a Rocky project and writes one VTK unstructured-grid
    file per time step.

    Args:
        rocky: Rocky API session (injected by
            :class:`~rocky_uniaxc.pyrocky.helpers.pyrocky_run`).
        rocky_filepath: Path to the Rocky project file (``.rocky``).
        output_dir: Directory where the VTK files will be saved.

    Raises:
        FileNotFoundError: If the Rocky project file does not exist.
        ValueError: If the file is not a valid ``.rocky`` project file.
    """
    if not os.path.isfile(rocky_filepath):
        raise FileNotFoundError(f"Rocky file not found: {rocky_filepath}")
    if not rocky_filepath.endswith(".rocky"):
        raise ValueError(f"Invalid Rocky file: {rocky_filepath}")

    os.makedirs(output_dir, exist_ok=True)

    tempdir = os.path.join(os.getcwd(), "pyrocky_temp")
    os.makedirs(tempdir, exist_ok=True)

    if not os.path.isfile(rocky_filepath):
        raise FileNotFoundError(f"Rocky file not found: {rocky_filepath}")
    project = rocky.api.OpenProject(rocky_filepath)
    study = project.GetStudy()

    particle_path = export_particle_stl(project, study)
    particles = study.GetParticles()

    timeset = study.GetTimeSet()
    for idx, t in enumerate(timeset):
        if particles.GetNumberOfParticles(time_step=idx) == 0:
            print(f"No particles at time {t:.2f}s, skipping VTK generation.")
            continue
        positions = np.vstack(
            [
                particles.GetGridFunction("Coordinate : X").GetArray(time_step=idx),
                particles.GetGridFunction("Coordinate : Y").GetArray(time_step=idx),
                particles.GetGridFunction("Coordinate : Z").GetArray(time_step=idx),
            ]
        ).transpose()
        orients = np.vstack(
            [
                particles.GetGridFunction("Orientation : Vector : X").GetArray(
                    time_step=idx
                ),
                particles.GetGridFunction("Orientation : Vector : Y").GetArray(
                    time_step=idx
                ),
                particles.GetGridFunction("Orientation : Vector : Z").GetArray(
                    time_step=idx
                ),
            ]
        ).transpose()
        trans_vels = np.vstack(
            [
                particles.GetGridFunction("Velocity : Translational : X").GetArray(
                    time_step=idx
                ),
                particles.GetGridFunction("Velocity : Translational : Y").GetArray(
                    time_step=idx
                ),
                particles.GetGridFunction("Velocity : Translational : Z").GetArray(
                    time_step=idx
                ),
            ]
        ).transpose()
        rotat_vels = np.vstack(
            [
                particles.GetGridFunction("Velocity : Rotational : X").GetArray(
                    time_step=idx
                ),
                particles.GetGridFunction("Velocity : Rotational : Y").GetArray(
                    time_step=idx
                ),
                particles.GetGridFunction("Velocity : Rotational : Z").GetArray(
                    time_step=idx
                ),
            ]
        ).transpose()
        residence_times = particles.GetGridFunction("Residence Time").GetArray(
            time_step=idx
        )
        output_vtk_path = os.path.join(
            output_dir,
            f"particles_t{t:.2f}",
        )

        _vtk_gen(
            stl_path=particle_path,
            positions=positions,
            orientations=orients,
            trans_vel_data=trans_vels,
            rotat_vel_data=rotat_vels,
            residence_times=residence_times,
            output_vtk_path=output_vtk_path,
        )
    project.CloseProject(check_save_state=False)
