# Utility functions for generating 2D/3D meshes using Gmsh
#
# Author: S.B. Lunowa

import gmsh
import numpy as np
import os

def spiral_centerline(radius: float, turns: float, length: float, N_seg: int,
                      x_shift: float=0, t_shift: float=0) -> np.ndarray:
    """
    Create points of a spiral centerline defined along the x-axis.

    Args:
        radius (float): Radius of the spiral.
        turns (float): Number of turns in the spiral.
        length (float): Length of the spiral along the cylinder axis.
        N_seg (int): Number of segments to discretize the spiral.
        x_shift (float): Optional x shift to apply to all points.
        t_shift (float): Optional phase shift in radians.

    Returns:
        np.ndarray: Array of shape (3, N_seg + 1) with x, y, z coordinates.
    """
    points = np.empty((3, N_seg + 1), dtype=float)
    points[0] = np.linspace(x_shift, x_shift + length, N_seg + 1)
    theta = np.linspace(t_shift, t_shift + 2 * np.pi * turns, N_seg + 1)
    points[1] = radius * np.cos(theta)
    points[2] = radius * np.sin(theta)
    return points

def compute_rotation(axis_from: np.ndarray, axis_to: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute the rotation axis and angle to get from axis_from to axis_to.

    Args:
        axis_from (np.ndarray): Starting axis (3D vector).
        axis_to (np.ndarray): Target axis (3D vector).

    Returns:
        tuple: (rotation_axis (np.ndarray), rotation_angle (float in radians))
    """
    axis_from = axis_from / np.linalg.norm(axis_from)
    axis_to = axis_to / np.linalg.norm(axis_to)
    angle = float(np.arccos(np.clip(np.dot(axis_from, axis_to), -1.0, 1.0)))
    axis = np.cross(axis_from, axis_to)
    return axis, angle


def create_spiral(centerline_points: np.ndarray, coil_radius: float, mesh_size: float=0) -> int:
    """
    Create a spiral coil volume in Gmsh using a spline representation (OpenCasCADE).

    Args:
        centerline_points (np.ndarray): Array of shape (3, N) with the spiral centerline points.
        coil_radius (float): Radius of the coil cross-section.
        mesh_size (float): Optional target (maximal) mesh size (for points on the centerline).

    Returns:
        int: Gmsh tag of the created coil volume.
    """
    point_tags = []
    for j in range(centerline_points.shape[1]):
        point_tags.append(gmsh.model.occ.addPoint(*centerline_points[:, j], meshSize=mesh_size))
    wire = gmsh.model.occ.addWire([gmsh.model.occ.addSpline(point_tags)])

    # Create coil volume by extruding disk along spline wire
    zaxis = centerline_points[:,1] - centerline_points[:,0]
    try:
        disk = gmsh.model.occ.addDisk(*centerline_points[:,0], coil_radius,coil_radius, zaxis=zaxis)
    except:
        # Fallback for older gmsh versions before 4.10
        disk = gmsh.model.occ.addDisk(*centerline_points[:,0], coil_radius,coil_radius)
        axis, angle = compute_rotation(np.array([0,0,1]), zaxis)
        if angle > 1e-12 and angle < np.pi - 1e-12:
            gmsh.model.occ.rotate([(2, disk)], *centerline_points[:,0], *axis, angle)
    dimtags = gmsh.model.occ.addPipe([(2, disk)], wire)
    coil = dimtags[0][1]
    gmsh.model.occ.remove([(2, disk)])
    gmsh.model.occ.synchronize()
    return coil


def create_cylinder_spiral(centerline_points: np.ndarray, radius: float, mesh_size: float=0) -> tuple[int, list[int]]:
    """
    Create a spiral coil surface in Gmsh using a union of cylinder pieces (geo).

    Args:
        centerline_points (np.ndarray): Array of shape (3, N) with the spiral centerline points.
        radius (float): Radius of the coil cross-section.
        mesh_size (float): Optional target (maximal) mesh size (for points on the centerline).

    Returns:
        int: tag of the created coil surface loop.
        list[int]: tags of the individual surfaces.
    """
    for i in range(centerline_points.shape[1]):
        point = centerline_points[:,i]
        center = gmsh.model.geo.addPoint(*point, mesh_size)
        if i == 0:
            n = centerline_points[:,i+1] - point
        elif i == centerline_points.shape[1] - 1:
            n = point - centerline_points[:,i-1]
        else:
            n = centerline_points[:,i+1] - centerline_points[:,i-1]
        n /= np.linalg.norm(n)
        if i == 0:
            lmbda, V = np.linalg.eigh(np.eye(3) - np.outer(n, n))
            if not np.allclose(lmbda, [0, 1, 1]):
                raise ValueError(f"The normal {n} is incompatible! Eigenvalues are {lmbda}")
        else:
            projection = np.eye(3) - np.outer(n, n)
            V = projection @ V_old
            V[:, 1] /= np.linalg.norm(V[:, 1])
            V[:, 2] /= np.linalg.norm(V[:, 2])
        circle_pts = [ point + radius * p for p in [V[:,1], V[:,2], -V[:,1], -V[:,2]] ]
        circle_pts = [ gmsh.model.geo.addPoint(*circle_pts[c], mesh_size) for c in range(4) ]
        arcs = [ gmsh.model.geo.addCircleArc(circle_pts[c], center, circle_pts[(c+1) % 4], -1, *n) for c in range(4) ]
        if i == 0:
            # add beginning circle
            loop = gmsh.model.geo.addCurveLoop(arcs)
            coil_parts = [ gmsh.model.geo.addPlaneSurface([loop]) ]
        else:
            # add sides of the cylinder
            lines = [ gmsh.model.geo.addLine(circle_pts[c], old_points[c]) for c in range(4) ]
            for c in range(4):
                loop = gmsh.model.geo.addCurveLoop([arcs[c], lines[(c+1) % 4], -old_arcs[c], -lines[c]])
                coil_parts.append(gmsh.model.geo.addSurfaceFilling([loop]))

        old_arcs = arcs
        old_points = circle_pts
        V_old = V

    # add closing circle
    loop = gmsh.model.geo.addCurveLoop(arcs)
    coil_parts.append( gmsh.model.geo.addPlaneSurface([loop]) )

    surface = gmsh.model.geo.addSurfaceLoop(coil_parts)
    gmsh.model.geo.synchronize()
    return surface, coil_parts


def set_meshing_options(mesh_size: float, n_curvature: int) -> None:
    """
    Set meshing options for Gmsh.

    Args:
        mesh_size (float): Target (maximal) mesh size.
        n_curvature (int): Number of elements per curvature radius.
    """
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", n_curvature)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.25 * mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)


def read_centerline_data(filename: str) -> list[np.ndarray]:
    """
    Read the centerline data from a OBJ file.

    Args:
        filename (str): The OBJ file to read

    Returns:
        list[np.ndarray]: List of arrays with shape (3, N) of points for each centerline.
    """
    points = []
    lines = []
    with open(filename) as f:
        while line := f.readline():
            if line.startswith("v "):
                parts = line.split()
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("l "):
                parts = line.split()
                line_points = [points[int(i)-1] for i in parts[1:]]
                lines.append(np.array(line_points).T)
            elif line[0] == '#':
                pass # ingoring comments
            else:
                print("Ignoring line: " + line)
    return lines


def write_centerlines_to_file(centerlines: list[np.ndarray], dir_name: str) -> None:
    """
    Write centerline points to an OBJ file.

    Args:
        centerlines (list[np.ndarray]): List of arrays with shape (3, N) of points for each centerline.
        dir_name (str): Directory to save the centerline file.
    """
    centerline_file = os.path.join(dir_name, "centerlines.obj")
    with open(centerline_file, "w") as f:
        Np = [0] * (len(centerlines)+1)
        for k, spiral_points in enumerate(centerlines):
            f.write(f"# vertices of spiral {k}\n")
            Np[k+1] = Np[k] + spiral_points.shape[1]
            for j in range(spiral_points.shape[1]):
                f.write(f"v {spiral_points[0,j]} {spiral_points[1,j]} {spiral_points[2,j]}\n")
        for k in range(len(centerlines)):
            f.write(f"# line for spiral {k}\nl ")
            indices = " ".join(str(j+1) for j in range(Np[k], Np[k+1]))
            f.write(indices + "\n")
    print(f"Wrote centerlines to \"{centerline_file}\".")


def write_mesh_of_current_gmsh_model(dir_name: str, name: str, formats: list[str]=["msh", "stl"], order: int=1) -> None:
    """
    Write the current gmsh model to mesh files.
    This creates both 2D and 3D mesh files `name_{N}d.{format}` (N = 2,3).

    Args:
        dir_name (str): Directory to save the mesh files.
        name (str): Base name for the mesh files.
        formats (list[str]): List of formats to write (default: ["msh", "stl"]).
        order (int): Mesh order (for higher-order meshes).
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if isinstance(formats, str):
        formats = [formats]

    for N in [2, 3]:
        generated = False
        for format in formats:
            if format == "stl" and N == 3:
                continue  # STL only defined for surfaces
            if not generated:
                gmsh.model.mesh.generate(N)
                if order > 1:
                    gmsh.model.mesh.setOrder(order)
            gmsh.write(os.path.join(dir_name, name + f"_{N}d.{format}"))
