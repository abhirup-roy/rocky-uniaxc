from ...packing3d import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
from matplotlib.colors import Normalize
from time import perf_counter


# To run the different examples, modify the if statement at the bottom of the
# script. There are three examples provided, each showcasing a different
# capability of Packing3D


def example1() -> None:
    # This example generates a cylindrical heatmap of packing densities
    # in a slice of a cylindrical container with loose packed particles in.

    file = r"post\particles_1700000.vtk"

    # Preload the dataset
    data = read_vtk_file(file)
    x_data, y_data, z_data, radii = retrieve_coordinates(data)

    # Convert cartesian position data from LIGGGHTS to cylindrical coordinates
    r_data, theta_data = convert_to_cylindrical(x_data, y_data)

    # Cylinder parameters
    cylinder_radius = 0.0375
    base_level = 0.005
    cylinder_height = 0.015
    r_divisions = 12
    theta_divisions = 3
    z_divisions = 1
    # radius_inner = 0.01
    radius_inner = None

    divisions = {"r": r_divisions,
                 "theta": theta_divisions,
                 "z": z_divisions}

    params = {"cylinder_radius": cylinder_radius,
              "cylinder_base_level": base_level,
              "cylinder_height": cylinder_height,
              "constant_volume": True}

    # Generate cylindrical mesh
    cylindrical_mesh = Mesh(system="cylindrical",
                            divisions=divisions,
                            **params)
    
    mesh_boundaries = cylindrical_mesh.cell_boundaries

    # Initialise a 3D array to store packing densities
    # Sparse dictionary representation for packing densities
    packing_densities = {}

    # Compute packing density for each cell
    for (i, j, k), boundaries in mesh_boundaries:
        packing_density = compute_packing_cylindrical(
            boundaries=boundaries,
            r_data=r_data,
            theta_data=theta_data,
            z_data=z_data,
            radii=radii,
            accurate_cylindrical=True
        )
        packing_densities[(i, j, k)] = packing_density


    # Access densities for a specific z-layer
    z_layer = 0
    # Plot using polar coordinates with wedges
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))

    # Define constants for FCC packing
    FCC_LIMIT = np.pi * np.sqrt(2) / 6  # Approx 0.7405
    FCC_HALF = FCC_LIMIT * (3/4)        # Approx 0.37025
    # Create a Normalize object with fixed limits
    norm = Normalize(vmin=FCC_HALF, vmax=FCC_LIMIT)

    # Iterate over mesh boundaries and add wedges
    for (i, j, k), boundaries in mesh_boundaries:
        if k == z_layer:
            # Extract boundaries from the mesh
            r_min = boundaries["r_min"]
            r_max = boundaries["r_max"]
            theta_min = boundaries["theta_min"]
            theta_max = boundaries["theta_max"]

            # Special case: Inner cell (0, 0, k)
            if (i, j) == (0, 0):
                # Override with appropriate ranges for plotting
                try:
                    radius_inner
                except NameError:
                    r_min, r_max = 0, cylinder_radius / r_divisions
                else:
                    if radius_inner is None:
                        r_min, r_max = 0, cylinder_radius / r_divisions
                    else:
                        r_min, r_max = 0, radius_inner

                theta_min, theta_max = 0, 2 * np.pi

            # Retrieve packing density for the cell
            density = packing_densities.get((i, j, k), np.nan)

            # Skip cells with no density (optional)
            if np.isnan(density):
                print("NaN found in calculated packing densities")
                print(f"Boundaries are: {boundaries}")
                continue

            # Create a wedge
            wedge = Wedge(
                center=(0, 0),  # Polar plot center
                r=r_max,
                theta1=np.degrees(theta_min),
                theta2=np.degrees(theta_max),
                width=r_max - r_min,  # Radial width
                transform=ax.transData._b,  # Apply polar transformation
                facecolor=plt.cm.plasma(norm(density)),  # Normalize density to fixed range
                edgecolor='none'
            )
            ax.add_patch(wedge)

    # Set limits
    ax.set_ylim(0, cylinder_radius)

    # Add a colorbar manually
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Packing Density")

    # Setup for radial axis
    r_tick_spacing = cylinder_radius / r_divisions
    r_ticks = np.arange(0, cylinder_radius, r_tick_spacing)
    ax.set_yticks(r_ticks)  # Sets radial tick marks
    # ax.set_yticklabels([f"{tick:.4f}" for tick in r_ticks])
    ax.set_rgrids([])  # Removes radial grid and labels
    # 

    # Disable the azimuthal axis
    ax.set_thetagrids([])  # Removes azimuthal grid and labels
    # ax.set_xticks([])  # Removes azimuthal tick marks


    z_min = base_level + z_layer * cylinder_height / z_divisions
    z_max = z_min + cylinder_height / z_divisions
    # Add a title
    plt.title(f"Packing Density Distribution in Region z=[{z_min:.3f}, {z_max:.3f}]\n(Random loose packed, before vibration)")
    plt.show()


def example2() -> None:
    # This example is a check for a manually formed vtk file for a simple cubic
    # lattice containing 41^3 particles. Both coordinate systems are used, 
    # and the boundaries of both can be modified to experiment with.

    file = r"post\centered_simple_cubic_spheres_68921.vtk"

    # Preload the dataset
    print("Loading file...")
    tic = perf_counter()
    data = read_vtk_file(file)
    toc = perf_counter()
    print("Successfully loaded file")
    x_data, y_data, z_data, radii = retrieve_coordinates(data)
    # coords = np.column_stack((x_data, y_data, z_data))
    # print(coords)

    # Convert to cylindrical coordinates
    r_data, theta_data = convert_to_cylindrical(x_data, y_data)

    a = 40*0.004

    boundaries_3d = {
        "x_min" : -a,
        "x_max" :  a,
        "y_min" : -a,
        "y_max" :  a,
        "z_min" :  -a*0.2,
        "z_max" :  a
    }

    packing_density_3d = compute_packing_cartesian(file=None,
                                                 boundaries=boundaries_3d,
                                                 x_data=x_data,
                                                 y_data=y_data,
                                                 z_data=z_data,
                                                 radii=radii,
                                                 cylinder_radius=None)

    
    boundaries_cyl = {
        "r_min"     :  -1,
        "r_max"     :  a,
        "theta_min" :  np.pi/2,
        "theta_max" :  2*np.pi,
        "z_min"     :  -a*0.2,
        "z_max"     :  a
    }

    packing_density_cyl = compute_packing_cylindrical(file=None,
                                                 boundaries=boundaries_cyl,
                                                 r_data=r_data,
                                                 theta_data=theta_data,
                                                 z_data=z_data,
                                                 radii=radii,
                                                 accurate_cylindrical=True)

    print(f"\n=== Packing Density Results ===")
    print(f"Cartesian Packing Density:    {packing_density_3d}")
    print(f"Cylindrical Packing Density:  {packing_density_cyl}")
    print(f"Simple Cubic Packing Density: {np.pi/6}")

    print(f"\n=== Benchmarking Information ===")
    print(f"Execution Time for compute_packing_cylindrical: {(toc-tic)*1000:.3f} ms")


def example3() -> None:
    # This example displays a slice of particles with a cartesian mesh which
    # approximates a cylindrical shape.


    file = r"post\particles_cooleffect.vtk"

    # Preload the dataset
    data = read_vtk_file(file)
    x_data, y_data, z_data, radii = retrieve_coordinates(data)

    # Cylinder parameters
    cylinder_radius = 0.06
    base_level = 0.104  # for cool effect
    cylinder_height = 0.01
    x_divisions = 24
    y_divisions = 24
    z_divisions = 1

    # Generate Cartesian mesh
    mesh_boundaries = generate_cartesian_mesh(
        x_divisions=x_divisions,
        y_divisions=y_divisions,
        z_divisions=z_divisions,
        boundaries=None,
        cylindrical_mesh_shape=True,
        radius=cylinder_radius,
        base_level=base_level,
        height=cylinder_height
    )

    # Initialize a 3D array to store packing densities
    packing_densities = np.full((x_divisions, y_divisions, z_divisions), np.nan)

    # Compute packing density for each valid cell
    # Compute packing density for each valid cell
    for (i, j, k), boundaries in mesh_boundaries:
        packing_density = compute_packing_cartesian(boundaries=boundaries,
                                                    x_data=x_data,
                                                    y_data=y_data,
                                                    z_data=z_data,
                                                    radii=radii,
                                                    cylinder_radius=cylinder_radius)
        packing_densities[i, j, k] = packing_density

    # Example: Slice the packing densities at a specific height division
    z_layer = 0
    z_layer_density = packing_densities[:, :, z_layer]

    # Define constants for FCC packing
    FCC_LIMIT = np.pi * np.sqrt(2) / 6  # Approx 0.7405
    FCC_HALF = FCC_LIMIT * (1/2)        # Approx 0.37025
    # Create a Normalize object with fixed limits
    norm = Normalize(vmin=FCC_HALF, vmax=FCC_LIMIT)

    # Create a heatmap of the packing density distribution at z_layer
    plt.imshow(z_layer_density, origin="lower", cmap="viridis", norm=norm, extent=[0, x_divisions, 0, y_divisions])
    plt.colorbar(label="Packing Density")

    # Create mask with circular cutout
    center_x, center_y = x_divisions / 2, y_divisions / 2
    mask_radius = cylinder_radius * x_divisions / (2 * cylinder_radius)

    # Define the outer rectangle
    rect_coords = [
        (0, 0), (x_divisions, 0), (x_divisions, y_divisions), (0, y_divisions), (0, 0)
    ]

    # Define the circle cutout
    theta = np.linspace(0, 2 * np.pi, 500)  # More points for smoother circle
    circle_x = center_x + mask_radius * np.cos(theta)
    circle_y = center_y + mask_radius * np.sin(theta)

    # Combine the rectangle and the cutout
    mask_coords = rect_coords + list(zip(circle_x[::-1], circle_y[::-1]))

    # Plot the mask as a white patch
    mask_patch = Polygon(mask_coords, closed=True, facecolor="white", edgecolor="none")
    ax = plt.gca()  # Get current axis
    ax.add_patch(mask_patch)

    z_min = base_level + z_layer * cylinder_height / z_divisions
    z_max = z_min + cylinder_height / z_divisions
    # Add a title
    plt.title(f"Packing Density Distribution in Region z=[{z_min:.3f}, {z_max:.3f}]")
    plt.xlabel("X Division")
    plt.ylabel("Y Division")
    plt.show()


if __name__ == "__main__":
    example1()