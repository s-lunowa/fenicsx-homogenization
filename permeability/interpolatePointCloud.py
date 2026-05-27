import pyvista as pv
import numpy as np


# =====================================================
#  — Single-field interpolation
# =====================================================
def interpolate_vtk_field(
    vtk_path,
    query_points,
    field_name,
    strategy="closest_point",
    radius=0.0,
    scale=1.0,
    translation=None
):
    """
    Interpolate a single scalar or vector field from a VTK dataset at given query points,
    with optional geometric transformation (scaling and translation) applied to the data points.

    Parameters
    ----------
    vtk_path : str
        Path to the VTK file.
    query_points : np.ndarray, shape (M, 3)
        Array of coordinates where the field should be interpolated.
    field_name : str
        Name of the field to interpolate (must exist in mesh.point_data).
    strategy : str, default="closest_point"
        Interpolation strategy ("closest_point", "mean", "null_value").
    radius : float, default=0.0
        Radius for local interpolation if applicable.
    scale : float, default=1.0
        Uniform scaling factor applied about the mesh centroid before interpolation.
    translation : array-like of shape (3,), optional
        Vector by which to translate all mesh points *after* scaling.

    Returns
    -------
    interpolated_values : np.ndarray
        Interpolated values (shape (M,) for scalars or (M, 3) for vector fields).
    """
    mesh = pv.read(vtk_path)

    if field_name not in mesh.point_data:
        raise ValueError(f"Field '{field_name}' not found. Available fields: {list(mesh.point_data.keys())}")

    # --- Apply scaling and translation to mesh points ---
    points = mesh.points.copy()
    center = np.mean(points, axis=0)

    # Scale about centroid
    points = (points - center) * scale + center

    # Apply translation (if provided)
    if translation is not None:
        translation = np.asarray(translation)
        if translation.shape != (3,):
            raise ValueError("Translation must be a 3D vector (x, y, z).")
        points += translation

    # Update mesh points
    mesh.points = points

    # --- Build the interpolation cloud ---
    cloud = pv.PolyData(mesh.points)
    cloud[field_name] = mesh.point_data[field_name]

    query_cloud = pv.PolyData(query_points)
    interpolated = query_cloud.interpolate(cloud, radius=radius, strategy=strategy)

    return interpolated[field_name]


# =====================================================
#  — Bounding box helper
# =====================================================
def get_vtk_bounding_box(vtk_path, scale=1.0):
    """
    Get the bounding box of a VTK dataset, optionally scaled about its centroid.
    """
    mesh = pv.read(vtk_path)
    bounds = np.array(mesh.bounds)  # (xmin, xmax, ymin, ymax, zmin, zmax)
    center = np.array(mesh.center)
    extents = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])

    scaled_half = 0.5 * scale * extents
    bbox = {
        "xmin": center[0] - scaled_half[0],
        "xmax": center[0] + scaled_half[0],
        "ymin": center[1] - scaled_half[1],
        "ymax": center[1] + scaled_half[1],
        "zmin": center[2] - scaled_half[2],
        "zmax": center[2] + scaled_half[2],
    }
    return bbox, center


# =====================================================
#  — Sampling + Multi-field plotting
# =====================================================
def sample_and_plot_all_fields(vtk_path, n_points=1000, scale=1.0):
    import matplotlib.pyplot as plt
    """
    Sample random points in the XY mid-plane of a VTK dataset,
    interpolate porosity and wire direction components,
    and plot them side-by-side in a 2x2 Matplotlib figure.
    """
    bbox, center = get_vtk_bounding_box(vtk_path, scale)
    slice_z = center[2]

    # --- Generate random sample points in the mid-plane ---
    x = np.random.uniform(bbox["xmin"], bbox["xmax"], n_points)
    y = np.random.uniform(bbox["ymin"], bbox["ymax"], n_points)
    z = np.full_like(x, slice_z)
    query_points = np.column_stack((x, y, z))

    # --- Interpolate the fields ---
    porosity = interpolate_vtk_field(vtk_path, query_points, "porosity")
    wire_dir = interpolate_vtk_field(vtk_path, query_points, "wireDirection")

    # --- Split vector field ---
    interp_results = {
        "porosity": porosity,
        "wireDir_x": wire_dir[:, 0],
        "wireDir_y": wire_dir[:, 1],
        "wireDir_z": wire_dir[:, 2],
    }

    # --- Plot results ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Interpolated Fields at Mid-Plane (z ≈ {slice_z:.3f})", fontsize=14, fontweight='bold')

    fields = [
        ("porosity", "Porosity"),
        ("wireDir_x", "WireDir X"),
        ("wireDir_y", "WireDir Y"),
        ("wireDir_z", "WireDir Z"),
    ]

    for ax, (key, title) in zip(axs.ravel(), fields):
        sc = ax.scatter(
            query_points[:, 0],
            query_points[:, 1],
            c=interp_results[key],
            cmap="viridis",
            s=20,
            alpha=0.9
        )
        Xb = [bbox["xmin"], bbox["xmax"], bbox["xmax"], bbox["xmin"], bbox["xmin"]]
        Yb = [bbox["ymin"], bbox["ymin"], bbox["ymax"], bbox["ymax"], bbox["ymin"]]
        ax.plot(Xb, Yb, 'k--', lw=1.2)
        ax.set_aspect('equal', 'box')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return query_points, interp_results


# =====================================================
#  — Test interpolation in 3D
# =====================================================
def test_interpolation(vtk_path="./output/Porosity.vtk"):
    bbox, center = get_vtk_bounding_box(vtk_path, scale=1.0)
    print("Bounding box:", bbox)

    # --- Sample random 3D points ---
    N = 1000
    x = np.random.uniform(bbox["xmin"], bbox["xmax"], N)
    y = np.random.uniform(bbox["ymin"], bbox["ymax"], N)
    z = np.random.uniform(bbox["zmin"], bbox["zmax"], N)
    query_points = np.column_stack((x, y, z))

    # --- Interpolate scalar and vector fields ---
    porosity_interp = interpolate_vtk_field(vtk_path, query_points, "porosity")
    wire_dir_interp = interpolate_vtk_field(vtk_path, query_points, "wireDirection")

    print("\nPorosity sample:", porosity_interp[:5])
    print("Wire direction sample:", wire_dir_interp[:5])


# =====================================================
#  — Test interpolation in 3D by point perturbation
# =====================================================
def test_interpolation_accuracy(vtk_path="./output/Porosity.vtk", epsilon=1e-3):
    """
    Test interpolation consistency by perturbing the original mesh points slightly
    and checking how close the interpolated values remain.

    Parameters
    ----------
    vtk_path : str
        Path to the VTK file.
    epsilon : float, optional, default=1e-3
        Perturbation magnitude (fraction of bounding box size).

    Prints
    ------
    Maximum absolute difference (∞-norm) for both scalar and vector fields.
    """
    print(f"\n=== Testing interpolation accuracy with epsilon = {epsilon:.2e} ===")

    # --- Load mesh and get points ---
    mesh = pv.read(vtk_path)
    points = mesh.points
    N = points.shape[0]

    # --- Get bounding box size for scale-based perturbation ---
    bounds = np.array(mesh.bounds)
    box_size = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    max_extent = np.linalg.norm(box_size)

    # --- Generate small perturbations ---
    perturbations = (np.random.rand(N, 3) - 0.5) * 2.0 * epsilon * max_extent
    perturbed_points = points + perturbations

    # --- Original values from file ---
    porosity_orig = mesh.point_data["porosity"]
    wire_dir_orig = mesh.point_data["wireDirection"]

    # --- Interpolated values at perturbed points ---
    porosity_interp = interpolate_vtk_field(vtk_path, perturbed_points, "porosity")
    wire_dir_interp = interpolate_vtk_field(vtk_path, perturbed_points, "wireDirection")

    # --- Compute infinity norms ---
    porosity_diff = np.max(np.abs(porosity_interp - porosity_orig))
    wire_dir_diff = np.max(np.linalg.norm(wire_dir_interp - wire_dir_orig, axis=1))

    print(f"Max |ΔPorosity| = {porosity_diff:.3e}")
    print(f"Max ||ΔWireDirection|| = {wire_dir_diff:.3e}")

    # --- Optional thresholds for quick sanity checks ---
    if porosity_diff < 1e-2 and wire_dir_diff < 1e-2:
        print(" Interpolation accuracy looks good.")
    else:
        print("  Large deviation detected — check epsilon or interpolation parameters.")


# =====================================================
#  — Test 2D cross-section plotting
# =====================================================
def test_plot_cross_sections(vtk_path="./output/Porosity.vtk", n_points=10000, scale=1.0):
    query_points, interp = sample_and_plot_all_fields(vtk_path, n_points=n_points, scale=scale)
    print("Sampled points shape:", query_points.shape)
    print("Porosity sample:", interp["porosity"][:5])
    print("WireDir_x sample:", interp["wireDir_x"][:5])


# =====================================================
#  — Main entry
# =====================================================
if __name__ == "__main__":
    vtk_path = "./permeability/data/spiral/outputs/grid_REVradius_0.3_Filter_box_BC_avg_Nonecloud.vtk"
    #test_interpolation_accuracy(vtk_path)
    #test_interpolation(vtk_path)
    test_plot_cross_sections(vtk_path)
