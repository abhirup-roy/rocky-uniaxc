# This module contains all the necessary functions that are unique to
# calculating packing density in cartesian coordinates

import numpy as np

from .geometry import (compute_cell_volume,
                       single_cap_intersection,
                       double_cap_intersection,
                       triple_cap_intersection)

from .utils import (compute_automatic_boundaries,
                    calculate_overlaps,
                    calculate_active_overlap_values,
                    is_inside_boundaries,
                    is_outside_boundaries,
                    centre_inside_boundaries,
                    convert_boundaries_dictionary)

from .mesh import Mesh

def compute_packing_cartesian(file=None, boundaries=None,
                              x_data=None, y_data=None, z_data=None, radii=None,
                              cylinder_radius=None,
                              compute_partial_volumes=True):
    """
    Compute the packing density of particles within a defined boundary region.

    Args:
        boundaries (dict): Dictionary defining the cuboidal region boundaries.
        report_progress (bool, optional): If True, prints progress during
                                          computation.
        x_data, y_data, z_data (np.ndarray): Preloaded x, y, z coordinates of
                                             particles.
        radii (np.ndarray): Preloaded radii of particles.

    Returns:
        float: The packing density as the fraction of volume occupied by
        particles.

    Raises:
        ValueError: If coordinates or radii are not provided.
    """

    # tic = time.perf_counter()

    if x_data is None or y_data is None or z_data is None or radii is None:
        raise ValueError("x_data, y_data, z_data, and radii must be provided.")

    # Determine boundaries
    if boundaries is None:
        boundaries = compute_automatic_boundaries(x_data=x_data, y_data=y_data,
                                                  z_data=z_data,
                                                  r_data=None, theta_data=None,
                                                  system="cartesian")
    else:
        boundaries = convert_boundaries_dictionary(boundaries,
                                                   system="cartesian")
    
    cell_volume = compute_cell_volume(boundaries=boundaries,
                                      system="cartesian",
                                      cylinder_radius=cylinder_radius)
    
    if not compute_partial_volumes:
        inside_boundaries_mask = centre_inside_boundaries(x_data=x_data,
                                                          y_data=y_data,
                                                          z_data=z_data,
                                                          boundaries=boundaries,
                                                          system="cartesian")
        
        # print(f"{len(radii[inside_boundaries_mask])} valid particles out of {len(r_data)}")
        
        inside_particle_volumes = (4/3) * np.pi * (radii[inside_boundaries_mask])**3

        total_particle_volume = np.sum(inside_particle_volumes)

        packing_density = total_particle_volume/cell_volume

        return packing_density

    # Calculate overlaps
    overlaps = calculate_overlaps(x_data=x_data, y_data=y_data, z_data=z_data,
                                  r_data=None, theta_data=None,
                                  radii=radii, boundaries=boundaries,
                                  factor=None,
                                  system="cartesian")

    # Determine active overlap values
    total_particles = len(radii)
    active_overlap_values = (
        calculate_active_overlap_values(total_particles, x_data=x_data,
                                        y_data=y_data, z_data=z_data,
                                        r_data=None, theta_data=None,
                                        boundaries=boundaries,
                                        overlaps=overlaps,
                                        system="cartesian")
                            )

    # Pre-compute and store full particle volumes
    full_particle_volumes = (4/3) * np.pi * (radii)**3

    # Get masks for particles completely inside or outside boundaries
    inside_mask = is_inside_boundaries(x_data=x_data, y_data=y_data,
                                       z_data=z_data,
                                       r_data=None, theta_data=None,
                                       boundaries=boundaries,
                                       radii=radii,
                                       factor=None,
                                       system="cartesian")
    
    outside_mask = is_outside_boundaries(x_data=x_data, y_data=y_data,
                                         z_data=z_data,
                                         r_data=None, theta_data=None,
                                         boundaries=boundaries,
                                         radii=radii,
                                         factor=None,
                                         system="cartesian")

    # Initialise volume
    total_particle_volume = 0.0

    # Add volumes for fully inside particles
    total_particle_volume += np.sum(full_particle_volumes[inside_mask])

    # Skip fully outside particles (0 volume contribution)

    # Loop only over particles which are neither outside nor inside
    neither_mask = ~(inside_mask | outside_mask)


    total_particle_volume += sum(calculate_particle_volume(
                                 i, radii, active_overlap_values)
                                 for i in np.where(neither_mask)[0]
                                 )

    packing_density = total_particle_volume/cell_volume

    return packing_density


def calculate_particle_volume(i, radii, active_overlap_values) -> float:
    """
    Calculate the volume contribution of a given particle.
    If the particle is fully within the boundaries, its entire volume is
    counted; likewise, if it is fully outside, no volume is counted.
    The main functionality lies between these two cases, where the volume of
    intersection between the particle and the boundary region is calculated.

    Args:
        i (int): Index of the particle being evaluated.
        total_particles (int): Total number of particles.
        radii (np.ndarray): Array of particle radii.
        inside_mask (np.ndarray): Boolean mask indicating particles completely
        inside the boundaries.
        outside_mask (np.ndarray): Boolean mask indicating particles completely
        outside the boundaries.
        full_particle_volumes (np.ndarray): Array of full particle volumes.
        active_overlap_values (np.ndarray): Array of overlap distances with
        boundaries.
        report_progress (bool): If True, prints progress during computation.

    Returns:
        float: The volume contribution of the particle.
    """

    # # Check if particle is completely within boundaries
    # if inside_mask[i]:
    #     return full_particle_volumes[i]  # Add full volume directly

    # # Check if particle is completely outside boundaries
    # # This check is necessary for particles which intersect the (infinite)
    # # boundary planes but are outside the actual boundary region
    # elif outside_mask[i]:
    #     return 0

    # Otherwise, calculate the partial volume that overlaps the boundary region
    # else:
    # Initialise partial volume at zero
    partial_volume = 0.0

    # Create array of overlap distances, filtering NaN values for
    # non-overlapping boundaries
    overlap_values = active_overlap_values[i][~np.isnan(
        active_overlap_values[i])]

    # Calculate the number of overlaps per particle
    number_of_overlaps = len(overlap_values)

    # Determine partial volume depending on the number of boundaries
    if number_of_overlaps == 1:
        partial_volume = single_cap_intersection(
            radii[i], overlap_values[0])

    elif number_of_overlaps == 2:
        partial_volume = double_cap_intersection(
            radii[i], overlap_values[0], overlap_values[1])

    elif number_of_overlaps == 3:
        partial_volume = triple_cap_intersection(
            radii[i], overlap_values[0], overlap_values[1],
            overlap_values[2])

    # Return calculated particle volume
    return partial_volume


def generate_cartesian_mesh(x_divisions, y_divisions, z_divisions,
                            boundaries=None, cylindrical_mesh_shape = False,
                            radius=None, base_level=None, height=None):
    """
    Generate a Cartesian mesh that can approximate a cylindrical region.

    Args:
        x_divisions (int): Number of divisions along the x-axis.
        y_divisions (int): Number of divisions along the y-axis.
        z_divisions (int): Number of divisions along the z-axis.
        boundaries (dict,optional): Major boundaries for the cartesian mesh
        cylindrical_mesh_shape (bool): Boolean that creates a cylindrical mesh
                                       with cartesian cells.
        radius (float, optional): Radius of the cylinder.
        base_level (float, optional): Base level of the cylinder in the
                                      z-direction.
        height (float, optional): Height of the cylinder.

    Returns:
        list: A list of tuples, each containing (indices, boundaries).
              - Indices: (i, j, k) for the cell location.
              - Boundaries: A dictionary of x, y, and z limits.
    """

    print("WARNING: Function generate_cartesian_mesh is now deprecated. Use Mesh class instead (see documentation).")

    # DEPRECATED FUNCTIONALITY # NOW USE MESH CLASS INSTEAD #

    # if cylindrical_mesh_shape:
    #     if radius is None or base_level is None or height is None:
    #         raise ValueError("""Cylinder data must be provided for
    #                          cylindrical_mesh_shape = True""")
    #     else:
    #         # Create grid points in x, y, and z directions
    #         x_bounds = np.linspace(-radius, radius, x_divisions + 1)
    #         y_bounds = np.linspace(-radius, radius, y_divisions + 1)
    #         z_bounds = np.linspace(base_level, base_level + height,
    #                             z_divisions + 1)
    # else:
    #     if boundaries is None:
    #         raise ValueError("""Boundaries must be specified for
    #                          cylindrical_mesh_shape = False or None""")
    #     else:
    #         x_min, x_max = boundaries["x_min"], boundaries["x_max"]
    #         y_min, y_max = boundaries["y_min"], boundaries["y_max"]
    #         z_min, z_max = boundaries["z_min"], boundaries["z_max"]
    #         x_bounds = np.linspace(x_min, x_max, x_divisions + 1)
    #         y_bounds = np.linspace(y_min, y_max, y_divisions + 1)
    #         z_bounds = np.linspace(z_min, z_max, z_divisions + 1)

    # mesh_boundaries = []

    # # Loop through Cartesian cells
    # for i in range(x_divisions):
    #     for j in range(y_divisions):
    #         for k in range(z_divisions):
    #             x_min, x_max = x_bounds[i], x_bounds[i + 1]
    #             y_min, y_max = y_bounds[j], y_bounds[j + 1]
    #             z_min, z_max = z_bounds[k], z_bounds[k + 1]

    #             if cylindrical_mesh_shape:
    #                 # Check if the center of the cell is inside the cylinder
    #                 corners = [
    #                     (x_min, y_min), (x_min, y_max),
    #                     (x_max, y_min), (x_max, y_max),
    #                 ]
    #                 distances = [np.sqrt(x**2 + y**2) for x, y in corners]
    #                 if not all(d >= radius for d in distances):
    #                     mesh_boundaries.append((
    #                         (i, j, k),  # Division indices
    #                         {
    #                             "x_min": x_min,
    #                             "x_max": x_max,
    #                             "y_min": y_min,
    #                             "y_max": y_max,
    #                             "z_min": z_min,
    #                             "z_max": z_max,
    #                         }
    #                     ))
    #             else:
    #                 mesh_boundaries.append((
    #                         (i, j, k),  # Division indices
    #                         {
    #                             "x_min": x_min,
    #                             "x_max": x_max,
    #                             "y_min": y_min,
    #                             "y_max": y_max,
    #                             "z_min": z_min,
    #                             "z_max": z_max,
    #                         }
    #                     ))

    # return mesh_boundaries

    divisions = {"x": x_divisions,
                 "y": y_divisions,
                 "z": z_divisions}

    params = {**boundaries,
              "cylinder_radius": radius,
              "cylinder_base_level": base_level,
              "cylinder_height": height,
              "cylindrical_mesh_shape": cylindrical_mesh_shape}

    # Generate Cartesian mesh
    cartesian_mesh = Mesh(system="cartesian",
                          divisions=divisions,
                          **params)
    
    mesh_boundaries = cartesian_mesh.cell_boundaries

    return list(enumerate(mesh_boundaries))
