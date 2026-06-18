"""Mesh generation for uniaxial compression simulations using GMSH.

Generates STL mesh files for the top compression wall, bottom wall, and
insert surface used in uniaxial compression tests.

Author:
    Abhirup Roy

Status:
    Development
"""

__author__ = "Abhirup Roy"
__email__ = "axr154@bham.ac.uk"
__status__ = "Development"


import os
import pathlib
import sys
import gmsh


def create_topwall(size, meshsize, out_dir):
    """Generate the top compression wall mesh and save as STL.

    Args:
        size: Side length of the square wall (m).
        meshsize: Desired mesh element size.
        out_dir: Directory to write ``compressive_wall1.stl`` into.
    """
    gmsh.model.add("wall1")
    p1 = gmsh.model.geo.addPoint(-size / 2, -size, -size / 2, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(size / 2, -size, -size / 2, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(-size / 2, -size, size / 2, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(size / 2, -size, size / 2, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p4)
    l3 = gmsh.model.geo.addLine(p4, p3)
    l4 = gmsh.model.geo.addLine(p3, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(out_dir, "compressive_wall1.stl"))
    gmsh.model.remove()  # Clear current model


def create_bottomwall(size, meshsize, out_dir):
    """Generate the bottom compression wall mesh and save as STL.

    Args:
        size: Side length of the square wall (m).
        meshsize: Desired mesh element size.
        out_dir: Directory to write ``compressive_wall2.stl`` into.
    """
    gmsh.model.add("wall2")
    p1 = gmsh.model.geo.addPoint(size / 2, size, -size / 2, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(-size / 2, size, -size / 2, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(size / 2, size, size / 2, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(-size / 2, size, size / 2, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p4)
    l3 = gmsh.model.geo.addLine(p4, p3)
    l4 = gmsh.model.geo.addLine(p3, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(out_dir, "compressive_wall2.stl"))
    gmsh.model.remove()


def create_insert(size, meshsize, out_dir):
    """Generate the insert surface mesh and save as STL.

    Args:
        size: Side length of the square insert (m).
        meshsize: Desired mesh element size.
        out_dir: Directory to write ``insert.stl`` into.
    """
    gmsh.model.add("insert")
    p1 = gmsh.model.geo.addPoint(-size / 2, -size / 2, size / 2, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(size / 2, -size / 2, size / 2, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(size / 2, size / 2, size / 2, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(-size / 2, size / 2, size / 2, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(out_dir, "insert.stl"))
    gmsh.model.remove()


def create_meshes(
    size: float, meshsize: float = 0.001, out_dir: str | pathlib.Path = "meshes"
) -> None:
    """Create all required meshes with GMSH and save them to disk.

    Generates three STL files — ``compressive_wall1.stl``,
    ``compressive_wall2.stl``, and ``insert.stl`` — in the output
    directory.

    Args:
        size: Side length of the walls and insert (m).
        meshsize: Desired mesh resolution. Defaults to 0.001.
        out_dir: Directory to save the meshes. Defaults to ``"meshes"``.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GMSH only once
    gmsh.initialize(sys.argv)

    # First wall
    create_topwall(size, meshsize, out_dir)
    # Second wall
    create_bottomwall(size, meshsize, out_dir)
    # Insert
    create_insert(size, meshsize, out_dir)

    # Finalize GMSH
    gmsh.finalize()
