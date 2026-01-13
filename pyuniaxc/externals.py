import os
import subprocess
import tempfile
from warnings import warn
import numpy as np
from stl import mesh
from evtk import vtk, hl
import ansys.rocky.core as pyrocky


def _get_rotation_matrix(vec: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix to align z-axis with the given vector.
    Using Rodrigues' rotation formula.
    """
    angle = np.linalg.norm(vec)

    axis = vec / angle
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def _load_particle_stl(file_path: str):
    """Load particle STL file and return vertices and faces."""
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


def _export_particle_stl(rocky_file: str) -> str:
    tempdir = tempfile.mkdtemp(prefix="rocky_particle")

    script_lines = [
        "import os",
        f"app.OpenProject(r'{rocky_file}')",
        "study = app.GetStudy()",
        "particle = study.GetParticleCollection()[0]",
        "export_toolkit = study.GetExportToolkit()",
        f"stl_path = os.path.join(r'{tempdir}', 'particle.stl')",
        "export_toolkit.ExportParticleToStl(",
        "    stl_filename=stl_path,",
        "    particle=particle,",
        "    time_to_export=-1",
        ")",
    ]
    script = "\n".join(script_lines)

    rocky_dir = os.path.dirname(rocky_file)
    write_path = os.path.join(rocky_dir, "export_particle_stl.py")
    with open(write_path, "w") as f:
        f.write(script)

    cmd = ["Rocky", "--script", write_path, "--headless"]
    subprocess.run(cmd, cwd=rocky_dir)
    stl_path = os.path.join(tempdir, "particle.stl")

    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Failed to export particle STL to {stl_path}")

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
    """
    Convert STL particle data and positions to VTK format.
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


def generate_vtk(
    rocky_filepath: str | list, rocky_exe: str = None, output_dir: str | list = None
):
    if not os.path.isfile(rocky_filepath):
        raise FileNotFoundError(f"Rocky file not found: {rocky_filepath}")
    if not rocky_filepath.endswith(".rocky"):
        raise ValueError(f"Invalid Rocky file: {rocky_filepath}")

    if not rocky_exe:
        rocky_exe = subprocess.run(["which", "Rocky"], capture_output=True)
        rocky_exe = rocky_exe.stdout.decode().strip()

    os.makedirs(output_dir, exist_ok=True)

    tempdir = os.path.join(os.getcwd(), "pyrocky_temp")
    os.makedirs(tempdir, exist_ok=True)
    rocky = pyrocky.launch_rocky(rocky_exe=rocky_exe)

    if isinstance(rocky_filepath, str):
        rocky_filepath = [rocky_filepath]
    if isinstance(output_dir, str):
        output_dir = [output_dir]

    if len(rocky_filepath) != len(output_dir):
        raise ValueError("rocky_filepath and output_dir must have the same length.")

    print(rocky_filepath)
    particle_paths = [
        _export_particle_stl(os.path.join(os.getcwd(), f)) for f in rocky_filepath
    ]

    for i, rocky_file in enumerate(rocky_filepath):
        if not os.path.isfile(rocky_file):
            raise FileNotFoundError(f"Rocky file not found: {rocky_file}")
        project = rocky.api.OpenProject(rocky_file)
        study = project.GetStudy()

        particles = study.GetParticles()
        particle = study.GetParticleCollection()[0]
        particle_stl_path = os.path.join(tempdir, "particle.stl")
        export_toolkit = study.GetExportToolkit()
        export_toolkit.ExportParticleToStl(
            stl_filename=particle_stl_path, particle=particle, time_to_export=-1
        )
        # if not os.path.exists(particle_stl_path):
        #     rocky.close()
        #     raise FileNotFoundError(
        #         f"Failed to export particle STL to {particle_stl_path}"
        #     )
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
                output_dir[i],
                f"particles_t{t:.2f}",
            )
            print(particle_paths)
            print(i)
            print(particle_paths[i])
            _vtk_gen(
                stl_path=particle_paths[i],
                positions=positions,
                orientations=orients,
                trans_vel_data=trans_vels,
                rotat_vel_data=rotat_vels,
                residence_times=residence_times,
                output_vtk_path=output_vtk_path,
            )
        project.CloseProject(check_save_state=False)
    print("tempfile dir: ", tempdir)
    rocky.close()
