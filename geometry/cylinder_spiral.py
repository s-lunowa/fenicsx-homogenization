# Generate 2D/3D meshes of a cylinder with two spiral coils inside using Gmsh
#
# Author: S.B. Lunowa

import argparse
import gmsh
import numpy as np
import os
import utils

if __name__ == "__main__":
    # create output directory
    dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cylinder_spiral")
    os.makedirs(dir_name, exist_ok=True)

    ######################################################################################
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--n_curvature", type=int, default=25, help="Number of elements per curvature radius.")
    parser.add_argument("-m", "--mesh_size", type=float, default=0.035, help="Target (maximal) mesh size.")
    parser.add_argument("-o", "--order", type=int, default=2, help="Mesh order for higher-order meshes (>= 1).")
    parser.add_argument("-r", "--radius", type=float, default=0.25, help="Radius of the coil cross-section.")
    parser.add_argument("-s", "--n_segments", type=int, default=300, help="Number of segments to discretize each spiral.")
    args = parser.parse_args()
    print(f"Options used: {args}", flush=True)

    cylinder_radius = 1.0        # radius of the cylinders
    total_length = 6             # length of extended cylinder
    cylinder_length = 3          # length of cylinder with coils
    # shift of extended cylinder along x-axis, i.e. cylinder with coils starts at x=0
    x_shift = -0.5*(total_length-cylinder_length)

    spiral_radii = [0.675, 0.25] # outer radii of the spirals
    spiral_t_shifts = [0, np.pi] # phase shifts of the spirals
    spiral_turns = 7             # number of turns of the spirals

    ######################################################################################
    # Create cylinders of length total_length and cylinder_length
    for length, shift, name in [(total_length, x_shift, "domain_extended"), (cylinder_length, 0, "domain")]:
        gmsh.initialize()
        gmsh.model.add(name)

        # Create cylinder: base center (x_shift,0,0), axis along x of length total_length
        volume = gmsh.model.occ.addCylinder(shift, 0, 0, length, 0, 0, cylinder_radius)
        gmsh.model.occ.synchronize()

        # define physical groups
        inlet = 3  # cylinder start
        outlet = 2 # cylinder end
        walls = [1] # cylinder walls and coil surfaces
        print(f"Inlet: {inlet}, outlet: {outlet}, walls: {walls}")

        try:
            gmsh.model.addPhysicalGroup(2, [inlet],  tag=1, name="inlet")
            gmsh.model.addPhysicalGroup(2, [outlet], tag=2, name="outlet")
            gmsh.model.addPhysicalGroup(2, walls,    tag=3, name="wall")
            gmsh.model.addPhysicalGroup(3, [volume], tag=1, name="mesh")
        except TypeError:
            # fallback for older gmsh versions
            gmsh.model.addPhysicalGroup(2, [inlet],  tag=1)
            gmsh.model.addPhysicalGroup(2, [outlet], tag=2)
            gmsh.model.addPhysicalGroup(2, walls,    tag=3)
            gmsh.model.addPhysicalGroup(3, [volume], tag=1)
        gmsh.model.occ.synchronize()

        # meshing
        utils.set_meshing_options(args.mesh_size, args.n_curvature)
        utils.write_mesh_of_current_gmsh_model(dir_name, name, order=args.order)
        gmsh.finalize()

    ######################################################################################
    # Create spirals only
    name = "coils"
    gmsh.initialize()
    gmsh.model.add(name)

    # lists of centerline points
    centerlines = []

    for spiral_radius, t_shift in zip(spiral_radii, spiral_t_shifts):
        # Create spiral along centerline points with given parameters
        points = utils.spiral_centerline(spiral_radius, spiral_turns, total_length, args.n_segments, x_shift, t_shift)
        centerlines.append(points)
        # create spiral coil volume
        utils.create_spiral(points, args.radius, args.mesh_size)
    gmsh.model.occ.synchronize()

    # save centerlines
    utils.write_centerlines_to_file(centerlines, dir_name)

    # meshing
    utils.set_meshing_options(args.mesh_size, args.n_curvature)
    utils.write_mesh_of_current_gmsh_model(dir_name, name, ["stl"], args.order)
    gmsh.finalize()

    ######################################################################################
    # Create cylinder with spirals inside
    name = "domain_cut"
    gmsh.initialize()
    gmsh.model.add(name)

    # Create cylinder: base center (0,0,0), axis along x
    volume = gmsh.model.occ.addCylinder(0, 0, 0, cylinder_length, 0, 0, cylinder_radius)

    for i, spiral_radius, t_shift in zip(range(len(centerlines)), spiral_radii, spiral_t_shifts):
        # Create spiral coil volume and remove from remaining cylinder volume
        coil = utils.create_spiral(centerlines[i], args.radius, args.mesh_size)
        dimtags = gmsh.model.occ.cut([(3, volume)], [(3, coil)], removeObject=True, removeTool=True)
        volume = dimtags[0][0][1]
        gmsh.model.occ.synchronize()

    # define physical groups
    inlet = 3  # cylinder start
    outlet = 2 # cylinder end
    walls = [1,4,5] # cylinder walls and coil surfaces
    print(f"Inlet: {inlet}, outlet: {outlet}, walls: {walls}")

    try:
        gmsh.model.addPhysicalGroup(2, [inlet],  tag=1, name="inlet")
        gmsh.model.addPhysicalGroup(2, [outlet], tag=2, name="outlet")
        gmsh.model.addPhysicalGroup(2, walls,    tag=3, name="wall")
        gmsh.model.addPhysicalGroup(3, [volume], tag=1, name="mesh")
    except TypeError:
        # fallback for older gmsh versions
        gmsh.model.addPhysicalGroup(2, [inlet],  tag=1)
        gmsh.model.addPhysicalGroup(2, [outlet], tag=2)
        gmsh.model.addPhysicalGroup(2, walls,    tag=3)
        gmsh.model.addPhysicalGroup(3, [volume], tag=1)
    gmsh.model.occ.synchronize()
    #gmsh.fltk.run()

    # meshing
    utils.set_meshing_options(args.mesh_size, args.n_curvature)
    utils.write_mesh_of_current_gmsh_model(dir_name, name, order=1) # ufl.CellDiameter does only allow 1st order
    gmsh.finalize()
