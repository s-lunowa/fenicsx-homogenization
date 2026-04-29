# real_case_validation.py

import os
import numpy as np
from math import acos, degrees

from interpolatePointCloud import interpolate_vtk_field
import homogenizationConfig as hc
from aneurysm import Aneurysm
from homogenizer import Homogenizer



def to_tex_decimal(x, eps=1e-12):
    """Plain decimal formatting with zero simplification."""
    if abs(x) < eps:
        return "0"
    return f"{x:.2f}".rstrip("0").rstrip(".")

def to_tex_scientific(x, eps=1e-12):
    """Scientific notation formatted for LaTeX: a.bc·10^{d}."""
    if abs(x) < eps:
        return "0"
    s = f"{x:.2e}"            # e.g. "-1.23e-04"
    base, exp = s.split('e')  # "-1.23", "-04"
    exp = int(exp)
    return rf"{base}\cdot 10^{{{exp}}}"

def latex_matrices(*matrix_specs):
    """
    Print multiple matrices inside ONE LaTeX environment.
    Each argument is a tuple: (matrix, scientific)
       matrix: numpy array
       scientific: True → exponential notation
                   False → plain decimal
    """
    print(r"\[")
    for M, sci in matrix_specs:
        print(r"\begin{pmatrix}")
        for i, row in enumerate(M):
            if sci:
                row_str = " & ".join(to_tex_scientific(x) for x in row)
            else:
                row_str = " & ".join(to_tex_decimal(x) for x in row)

            end = r" \\" if i < M.shape[0] - 1 else ""
            print(f"  {row_str}{end}")
        print(r"\end{pmatrix}")
    print(r"\]")


# ----------------------------------------------------------
# Helper: angle between two *lines* (sign does not matter)
# ----------------------------------------------------------
def line_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Return minimum angle (0–90°) between two line directions."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosang = np.dot(a, b)
    cosang = np.clip(cosang, -1.0, 1.0)
    return degrees(acos(abs(cosang)))   # abs → line orientation (± irrelevant)


# ----------------------------------------------------------
# MAIN FUNCTION that you import and call
# ----------------------------------------------------------
def realCaseValidation(N: int = 40) -> None:
    """
    Performs:
      - homogenization of real-case aneurysm
      - sampling of 12 validation REVs
      - comparison against ground truth porosity & direction
    """

    # -------------------------------
    # Paths and setup
    # -------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir   = os.path.join(current_dir, "..", "geometry", "real_case")
    data_dir   = os.path.join(current_dir, "data", "real_case")
    os.makedirs(data_dir, exist_ok=True)

    coils = ["coil1.obj", "coil2.obj"]
    CLFilePaths = ["spline_1.obj", "spline_2.obj"]

    coilFilePath   = [os.path.join(mesh_dir, coil) for coil in coils]
    aneuFilePath   = os.path.join(mesh_dir, "aneurysm.obj")
    vesselFilePath = os.path.join(mesh_dir, "vessel.obj")
    CLFilePath     = [os.path.join(mesh_dir, spline) for spline in CLFilePaths]

    vtk_cloud_path = os.path.join(data_dir, "comparison_analytical_real_case_cloud.vtk")

    # -------------------------------
    # Generate homogenization field
    # -------------------------------
    print("\n>>> Running homogenization…")

    aneurysm = Aneurysm(coilFilePath, aneuFilePath, vesselFilePath, CLFilePath)
    aneurysm.set_sampling_params(
        scan_count=hc.SCAN_COUNT,
        scan_resolution=hc.SCAN_RESOLUTION,
        sample_point_count=hc.SAMPLE_POINT_COUNT,
        normal_sample_count=hc.NORMAL_SAMPLE_POINT_COUNT
    )

    aneurysm.generate_grid_and_masks(
        N=N,
        r_REV=0.5,
        padding_distance=0.5
    )

    homogenizer = Homogenizer(
        aneurysm,
        filtertype="box",
        volume_fraction_wall=0.0,
        wall_boundary_condition="const"
    )

    homogenizer.homogenize_and_output(
        os.path.join(data_dir, "comparison_analytical_real_case")
    )

    # -------------------------------
    # Validation REV center points
    # -------------------------------
    coords = np.array([
        [6.008,   6.0037, 3.4961],
        [8.000,   3.8000, 3.0000],
        [7.000,   6.5000, 4.0000],
        [5.670,   6.8200, 2.6000],
        [7.4735,  6.2620, 3.9615],
        [8.3517,  5.2054, 6.6000],
        [5.680,   7.2700, 2.6100],
        [6.144,   7.7200, 3.4894],
        [9.1149,  6.9253, 3.5424],
        [6.9426,  7.1500, 3.6673],
        [9.2000,  4.0700, 3.4500],
        [7.5000,  7.1100, 1.9800],
        [7.2000,  5.0000, 2.8000],
        [7.8900,  3.9700, 3.9800]
    ])

    neg_coords = -coords

    # -------------------------------
    # Interpolate porosity & direction
    # -------------------------------
    values = interpolate_vtk_field(
        vtk_cloud_path,
        neg_coords,
        "porosity",
        strategy="closest_point",
        radius=0.0
    )

    vectors = interpolate_vtk_field(
        vtk_cloud_path,
        neg_coords,
        "wireDirection",
        strategy="closest_point",
        radius=0.0
    )

    shape_tensor_normals_1 = interpolate_vtk_field(
        vtk_cloud_path,
        neg_coords,
        "shape_tensor_normal1",
        strategy="closest_point",
        radius=0.0
    )

    shape_tensor_normals_2 = interpolate_vtk_field(
        vtk_cloud_path,
        neg_coords,
        "shape_tensor_normal2",
        strategy="closest_point",
        radius=0.0
    )

    shape_tensor_eigenvalues = interpolate_vtk_field(
        vtk_cloud_path,
        neg_coords,
        "shape_tensor_eigenvalues",
        strategy="closest_point",
        radius=0.0
    )

    R_phis = np.zeros(values.shape + (3,3))
    assign = {0:"x", 1:"y", 2:"z"}
    for i in range(3):
        for j in range(3):
            R_phis[:,i,j] = interpolate_vtk_field(
                                        vtk_cloud_path,
                                        neg_coords,
                                        f"R_{assign[i]}{assign[j]}",
                                        strategy="closest_point",
                                        radius=0.0
                                    )
    K_par_effs = interpolate_vtk_field(
        vtk_cloud_path,
        neg_coords,
        "K_par_eff",
        strategy="closest_point",
        radius=0.0
    )

    K_perp_effs = interpolate_vtk_field(
        vtk_cloud_path,
        neg_coords,
        "K_perp_eff",
        strategy="closest_point",
        radius=0.0
    )

    # -------------------------------
    # Reference values (already y,z swapped then new y set to -y)
    # -------------------------------
    referenceValues = {
    1:  (0.6298, [-0.10,  0.65,  0.75], 3.25e-9),
    2:  (0.8866, [-1.00, -0.03,  0.01], 1.06e-7),
    3:  (0.7255, [-0.98,  0.13,  0.16], 6.95e-9),
    4:  (0.6944, [ 0.09, -0.04, -0.99], 2.35e-8),
    5:  (0.7031, [-0.68,  0.01,  0.73], 3.52e-9),
    6:  (0.6455, [-0.97,  0.15, -0.23], 1.00e-8),
    7:  (0.7010, [-0.33,  0.03,  0.94], 1.75e-8),
    8:  (0.6819, [-0.90, -0.16,  0.40], 7.73e-9),
    9:  (0.6915, [-0.58, -0.82,  0.03], 8.61e-9),
    10: (0.7472, [ 0.06, -0.36,  0.93], 7.59e-9),
    11: (0.9123, [-0.81, -0.53,  0.25], 8.20e-8),
    12: (0.9059, [-0.06,  0.00, -1.00], 9.79e-8),
    13: (0.8362, [-0.06,  0.00, -1.00], 9.79e-8), # is just a dummy line since we have not calculated this value
    14: (0.7701, [-0.06,  0.00, -1.00], 9.79e-8), # is just a dummy line since we have not calculated this value
}

    # -------------------------------
    # Comparison loop
    # -------------------------------
    print("\n>>> VALIDATION RESULTS\n")

    for i, (value, vector,shape_tensor_normal_1, shape_tensor_normal_2, eigenvalues, R_phi, K_par_eff, K_perp_eff) in enumerate(zip(values, vectors, shape_tensor_normals_1, shape_tensor_normals_2,shape_tensor_eigenvalues, R_phis, K_par_effs, K_perp_effs), start=1):
        ref_poro, ref_vec, ref_eig = referenceValues[i]

        # Porosity error
        poro_err = 100 * (value - ref_poro) / ref_poro

        R = np.array([shape_tensor_normal_1,shape_tensor_normal_2,vector]).T
        # swap the z with -y since blender was doing this also in this geometry
        S = np.array([[1,  0,  0], [0,  0,  1],[0, -1,  0]])
        R = S@R
        Eig = np.eye(3)
        Eig[0,0] = eigenvalues[0]
        Eig[1,1] = eigenvalues[1]
        Eig[2,2] = eigenvalues[2]

        # Direction error
        v = R[:,-1]
        r = np.array(ref_vec, dtype=float)
        angle = line_angle(v, r)



        print(f"REV {i:2d}:")
        print(f"  porosity pred = {value: .6f},   ref = {ref_poro: .6f},   error = {poro_err: .3f}%")
        print(f"  direction pred = ({v[0]: .4f}, {v[1]: .4f}, {v[2]: .4f})")
        print(f"  direction ref  = ({r[0]: .4f}, {r[1]: .4f}, {r[2]: .4f})")
        print(f"  direction error (angle) = {angle: .3f} degrees\n")

        print(f"{R}")
        print(f"{Eig}")

        latex_matrices((R,False),(Eig,True),(R.T,False))

        S1 = np.array([[0,  0,  1], [0,  1,  0],[1, 0,  0]]) #swap these wrt. the paper
        K = np.eye(3)
        K[0,0] = K_perp_eff
        K[1,1] = K_perp_eff
        K[2,2] = K_par_eff

        latex_matrices( ((S@R_phi@S1),False) ,(K,True),((S@R_phi@S1).T,False) )

        print(np.linalg.det(S@R_phi@S1))