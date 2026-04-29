# Generate a 3D mesh of a cylinder with a coil inside using Gmsh
#
# Author: S.B. Lunowa

import gmsh
import numpy as np
import os

import utils

def add_vessel_geometry(filename: str) -> tuple[int, list[int]]:
    """
    Adds the vessel geometry to the current Gmsh model.

    Returns:
        int: tag of the added surface loop
        list[int]: individual tags of the surfaces
    """

    gmsh.merge(filename)
    # Angle between two triangles above which an edge is considered as sharp
    angle = np.pi/4
    # For complex geometries, patches can be too complex, too elongated or too
    # large to be parametrized; setting the following option will force the
    # creation of patches that are amenable to reparametrization:
    forceParametrizablePatches = False
    # For open surfaces include the boundary edges in the classification process
    includeBoundary = True

    gmsh.model.mesh.classifySurfaces(angle, includeBoundary,
                                     forceParametrizablePatches)

    # Create a geometry for all the discrete curves and surfaces in the mesh, by
    # computing a parametrization for each one
    gmsh.model.mesh.createGeometry()

    # Create a volume from all the surfaces
    surface_tags = [dimtag[1] for dimtag in gmsh.model.getEntities(2)]
    surface = gmsh.model.geo.addSurfaceLoop(surface_tags)
    gmsh.model.geo.synchronize()
    return surface, surface_tags


def generateArtificialVesselMesh() -> None:
    dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artificial_vessel")

    # Parameters
    spline_files = [os.path.join(dir_name, f"spline_{i}.obj") for i in [1, 2]]
    scaling = 0.75
    radii = [scaling * 0.1778, scaling * 0.127]
    mesh_size = 0.16
    n_curvature = 20
    order = 1

    ######################################################################################
    # Create vessel of original and reduced length
    for name, filename in [("domain_extended", "vessel.stl"), ("domain", "aneurysm.stl")]:
        gmsh.initialize()
        gmsh.model.add(name)

        surface, surface_tags = add_vessel_geometry(os.path.join(dir_name, filename))
        volume = gmsh.model.geo.addVolume([surface])
        gmsh.model.geo.synchronize()

        # define physical groups
        inlet = 3
        outlet = 4
        walls = []
        for id in surface_tags:
            if id != inlet and id != outlet:
                walls.append(id)
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

        # meshing
        utils.set_meshing_options(mesh_size, n_curvature)
        utils.write_mesh_of_current_gmsh_model(dir_name, name, order=order)
        gmsh.finalize()

    ######################################################################################
    # Create spirals only
    name = "coils"
    gmsh.initialize()
    gmsh.model.add(name)

    # lists of centerline points
    centerlines = []

    for filename in spline_files:
        centerlines.extend(utils.read_centerline_data(filename))

    for centerline, radius in zip(centerlines, radii):
        utils.create_cylinder_spiral(centerline, radius, mesh_size)

    # save centerlines
    utils.write_centerlines_to_file(centerlines, dir_name)

    # meshing
    utils.set_meshing_options(mesh_size, n_curvature)
    utils.write_mesh_of_current_gmsh_model(dir_name, name, ["stl"], order)
    gmsh.finalize()

    ######################################################################################
    # Create cylinder with spirals inside
    name = "domain_cut"
    gmsh.initialize()
    gmsh.model.add(name)

    surf, surf_tags = add_vessel_geometry(os.path.join(dir_name, "aneurysm.stl"))
    surfaces = [ surf ]
    surface_tags = surf_tags

    for centerline, radius in zip(centerlines, radii):
        print(np.min(centerline, 1), np.max(centerline, 1))
        surf, surf_tags = utils.create_cylinder_spiral(centerline, radius, mesh_size)
        surfaces.append( surf )
        surface_tags.extend( surf_tags )

    gmsh.model.geo.addVolume(surfaces)
    gmsh.model.geo.synchronize()

    # define physical groups
    inlet = 3
    outlet = 4
    walls = []
    for id in surface_tags:
        if id != inlet and id != outlet:
            walls.append(id)
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

    #gmsh.fltk.run()

    # meshing
    utils.set_meshing_options(mesh_size, n_curvature)
    utils.write_mesh_of_current_gmsh_model(dir_name, name, order=1) # ufl.CellDiameter does only allow 1st order
    gmsh.finalize()

if __name__ == "__main__":
    generateArtificialVesselMesh()