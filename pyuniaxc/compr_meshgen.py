#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Abhirup Roy"
__email__ = "axr154@bham.ac.uk"
__status__ = "Development"

"""
This script generates meshes for Uniaxial Copmressions.
"""


import os
import sys
import gmsh


def create_particlebox(size, meshsize=0.001, gui=False):
    gmsh.initialize(sys.argv)
    gmsh.model.add("particlebox")

    # Define the half-size of the cube
    half_size = size / 2

    # Define the points for the cube
    p1 = gmsh.model.geo.addPoint(-half_size, -half_size, -half_size, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(half_size, -half_size, -half_size, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(half_size, half_size, -half_size, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(-half_size, half_size, -half_size, meshSize=meshsize)
    p5 = gmsh.model.geo.addPoint(-half_size, -half_size, half_size, meshSize=meshsize)
    p6 = gmsh.model.geo.addPoint(half_size, -half_size, half_size, meshSize=meshsize)
    p7 = gmsh.model.geo.addPoint(half_size, half_size, half_size, meshSize=meshsize)
    p8 = gmsh.model.geo.addPoint(-half_size, half_size, half_size, meshSize=meshsize)

    # Define the lines for the cube
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    l5 = gmsh.model.geo.addLine(p1, p5)
    l6 = gmsh.model.geo.addLine(p2, p6)
    l7 = gmsh.model.geo.addLine(p3, p7)
    l8 = gmsh.model.geo.addLine(p4, p8)
    l9 = gmsh.model.geo.addLine(p5, p6)
    l10 = gmsh.model.geo.addLine(p6, p7)
    l11 = gmsh.model.geo.addLine(p7, p8)
    l12 = gmsh.model.geo.addLine(p8, p5)

    # Define the surfaces for the cube (excluding the top face)
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    s2 = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l1, l6, -l9, -l5])]
    )
    s3 = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l2, l7, -l10, -l6])]
    )
    s4 = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l3, l8, -l11, -l7])]
    )
    s5 = gmsh.model.geo.addPlaneSurface(
        [gmsh.model.geo.addCurveLoop([l4, l5, -l12, -l8])]
    )

    # Create a volume from the surfaces
    gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    if gui and "-nopopup" not in sys.argv:
        gmsh.fltk.run()
    # Save the mesh to a file
    gmsh.write(os.path.join("meshes", "particlebox.stl"))
    gmsh.finalize()


def create_compr_walls(size, meshsize=0.001, gui=False):
    half_size = size / 2

    def wall1():
        gmsh.initialize(sys.argv)
        gmsh.model.add("compressive_walls")

        p1 = gmsh.model.geo.addPoint(-half_size, -size, -half_size, meshSize=meshsize)
        p2 = gmsh.model.geo.addPoint(half_size, -size, -half_size, meshSize=meshsize)
        p3 = gmsh.model.geo.addPoint(-half_size, -size, half_size, meshSize=meshsize)
        p4 = gmsh.model.geo.addPoint(half_size, -size, half_size, meshSize=meshsize)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p4)
        l3 = gmsh.model.geo.addLine(p4, p3)
        l4 = gmsh.model.geo.addLine(p3, p1)

        gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        if gui and "-nopopup" not in sys.argv:
            gmsh.fltk.run()
        # Save the mesh to a file
        gmsh.write(os.path.join("meshes", "compressive_wall1.stl"))

        gmsh.finalize()

    def wall2():
        gmsh.initialize(sys.argv)
        p1 = gmsh.model.geo.addPoint(half_size, size, -half_size, meshSize=meshsize)
        p2 = gmsh.model.geo.addPoint(-half_size, size, -half_size, meshSize=meshsize)
        p3 = gmsh.model.geo.addPoint(half_size, size, half_size, meshSize=meshsize)
        p4 = gmsh.model.geo.addPoint(-half_size, size, half_size, meshSize=meshsize)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p4)
        l3 = gmsh.model.geo.addLine(p4, p3)
        l4 = gmsh.model.geo.addLine(p3, p1)
        gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        # Save the mesh to a file
        gmsh.write(os.path.join("meshes", "compressive_wall2.stl"))
        if gui and "-nopopup" not in sys.argv:
            gmsh.fltk.run()
        gmsh.finalize()

    wall1()
    wall2()


def create_insert(size, meshsize=0.001, gui=False):
    gmsh.initialize(sys.argv)
    gmsh.model.add("insert")

    # Define the half-size of the cube
    half_size = size / 2

    # Define the points for the cube
    p1 = gmsh.model.geo.addPoint(-half_size, -half_size, half_size, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(half_size, -half_size, half_size, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(half_size, half_size, half_size, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(-half_size, half_size, half_size, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    if gui and "-nopopup" not in sys.argv:
        gmsh.fltk.run()
    # Save the mesh to a file
    gmsh.write(os.path.join("meshes", "insert.stl"))
    gmsh.finalize()


def create_meshes_efficiently(size, meshsize=0.001, out_dir="meshes"):
    """Create all required meshes with a single GMSH instance."""
    os.makedirs(out_dir, exist_ok=True)
    half_size = size / 2

    # Initialize GMSH only once
    gmsh.initialize(sys.argv)

    # First wall
    gmsh.model.add("wall1")
    p1 = gmsh.model.geo.addPoint(-half_size, -size, -half_size, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(half_size, -size, -half_size, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(-half_size, -size, half_size, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(half_size, -size, half_size, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p4)
    l3 = gmsh.model.geo.addLine(p4, p3)
    l4 = gmsh.model.geo.addLine(p3, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(out_dir, "compressive_wall1.stl"))
    gmsh.model.remove()  # Clear current model

    # Second wall
    gmsh.model.add("wall2")
    p1 = gmsh.model.geo.addPoint(half_size, size, -half_size, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(-half_size, size, -half_size, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(half_size, size, half_size, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(-half_size, size, half_size, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p4)
    l3 = gmsh.model.geo.addLine(p4, p3)
    l4 = gmsh.model.geo.addLine(p3, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(out_dir, "compressive_wall2.stl"))
    gmsh.model.remove()

    # Insert
    gmsh.model.add("insert")
    p1 = gmsh.model.geo.addPoint(-half_size, -half_size, half_size, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(half_size, -half_size, half_size, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(half_size, half_size, half_size, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(-half_size, half_size, half_size, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(out_dir, "insert.stl"))

    # Finalize GMSH
    gmsh.finalize()


# if __name__ == "__main__":
#     create_particlebox(0.1, meshsize=0.001)
#     create_compr_walls(0.1, meshsize=0.001)
#     create_insert(0.1, meshsize=0.001)
