from ast import List
import os
import numpy as np

from interpolatePointCloud import interpolate_vtk_field
import homogenizationConfig as hc
from aneurysm import Aneurysm
from homogenizer import Homogenizer
from rev_data import REVData


def relative_frobenius_error(A, B):
    """Compute relative error between two matrices in Frobenius norm."""
    return np.linalg.norm(A - B, ord="fro") / np.linalg.norm(A, ord="fro")

def permeability_tensor_error(R_pred, K_ref):
    eigenvalues_A, _ = np.linalg.eigh(K_ref)
    Keff = np.diag(eigenvalues_A)
    Kalgo_raw = R_pred @ Keff @ R_pred.T
    return relative_frobenius_error(Kalgo_raw, K_ref)

def rotation_vector_to_matrix(omega):
    """
    Convert a rotation vector (axis * angle, in radians) to a 3×3 rotation matrix using Rodrigues' formula.
    """
    def skew(omega):
        wx, wy, wz = omega
        return np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ])
    eta = np.linalg.norm(omega)

    if eta < 1e-12:
        return np.eye(3)

    K = skew(omega / eta)
    return (
        np.eye(3)
        + np.sin(eta) * K
        + (1.0 - np.cos(eta)) * (K @ K)
    )
    
def get_random_unit_direction(eta):
    # random unit direction
    u = np.random.randn(3)
    u = u / np.linalg.norm(u)
    omega = eta * u  # small rotation vector
    return rotation_vector_to_matrix(omega)

def perturbation_analysis(N: int = 100, etas: list = [1e-3, 1e-2, 1e-1], Rs = None, Keff: np.ndarray = None):
    
    mean_eta_errors = [[] for _ in Rs]
    std_eta_errors = [[] for _ in Rs]

    for eta in etas:

        errs_eta = [[] for _ in Rs]

        for _ in range(N):

            R_pert = get_random_unit_direction(eta)

            for i, R in enumerate(Rs):

                R_perturbed = R @ R_pert

                err = relative_frobenius_error(
                    R_perturbed @ Keff @ R_perturbed.T,
                    R @ Keff @ R.T
                )

                errs_eta[i].append(err)

        for i in range(len(Rs)):
            mean_eta_errors[i].append(np.mean(errs_eta[i]))
            std_eta_errors[i].append(np.std(errs_eta[i]))

    return mean_eta_errors, std_eta_errors

def run_real_case_homogenization(N: int = 40) -> str:
    """
    Run homogenization for the real-case aneurysm.

    Returns
    -------
    vtk_cloud_path : str
        Path to the generated VTK cloud file.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(current_dir, "..", "geometry", "real_case")
    data_dir = os.path.join(current_dir, "data", "real_case")
    os.makedirs(data_dir, exist_ok=True)

    coils = ["coil1.obj", "coil2.obj"]
    CLFilePaths = ["spline_1.obj", "spline_2.obj"]

    coilFilePath = [os.path.join(mesh_dir, coil) for coil in coils]
    aneuFilePath = os.path.join(mesh_dir, "aneurysm.obj")
    vesselFilePath = os.path.join(mesh_dir, "vessel.obj")
    CLFilePath = [os.path.join(mesh_dir, spline) for spline in CLFilePaths]

    output_base_path = os.path.join(data_dir, "comparison_analytical_real_case")
    vtk_cloud_path = output_base_path + "_cloud.vtk"

    print("\n>>> Running homogenization…")

    aneurysm = Aneurysm(
        coilFilePath,
        aneuFilePath,
        vesselFilePath,
        CLFilePath,
    )

    aneurysm.set_sampling_params(
        scan_count=hc.SCAN_COUNT,
        scan_resolution=hc.SCAN_RESOLUTION,
        sample_point_count=hc.SAMPLE_POINT_COUNT,
        normal_sample_count=hc.NORMAL_SAMPLE_POINT_COUNT,
    )

    aneurysm.generate_grid_and_masks(
        N=N,
        r_REV=0.5,
        padding_distance=0.5,
    )

    homogenizer = Homogenizer(
        aneurysm,
        filtertype="box",
        volume_fraction_wall=0.0,
        wall_boundary_condition="const",
    )

    homogenizer.homogenize_and_output(output_base_path)

    return vtk_cloud_path

def interpolate_real_case_to_revs(vtk_cloud_path: str) -> dict:
    """
    Interpolate homogenized fields to the validation REV center points.
    """

    revData = REVData()
    coords = revData.get_REV_positions()

    values = interpolate_vtk_field(
        vtk_cloud_path,
        coords,
        "porosity",
        strategy="closest_point",
        radius=0.0,
    )

    vectors = interpolate_vtk_field(
        vtk_cloud_path,
        coords,
        "wireDirection",
        strategy="closest_point",
        radius=0.0,
    )

    shape_tensor_normals_1 = interpolate_vtk_field(
        vtk_cloud_path,
        coords,
        "shape_tensor_normal1",
        strategy="closest_point",
        radius=0.0,
    )

    shape_tensor_normals_2 = interpolate_vtk_field(
        vtk_cloud_path,
        coords,
        "shape_tensor_normal2",
        strategy="closest_point",
        radius=0.0,
    )

    shape_tensor_eigenvalues = interpolate_vtk_field(
        vtk_cloud_path,
        coords,
        "shape_tensor_eigenvalues",
        strategy="closest_point",
        radius=0.0,
    )

    R_phis = np.zeros(values.shape + (3, 3))
    assign = {0: "x", 1: "y", 2: "z"}

    for i in range(3):
        for j in range(3):
            R_phis[:, i, j] = interpolate_vtk_field(
                vtk_cloud_path,
                coords,
                f"R_{assign[i]}{assign[j]}",
                strategy="closest_point",
                radius=0.0,
            )

    K_par_effs = interpolate_vtk_field(
        vtk_cloud_path,
        coords,
        "K_par_eff",
        strategy="closest_point",
        radius=0.0,
    )

    K_perp_effs = interpolate_vtk_field(
        vtk_cloud_path,
        coords,
        "K_perp_eff",
        strategy="closest_point",
        radius=0.0,
    )

    return {
        "coords": coords,
        "porosity": values,
        "wireDirection": vectors,
        "shape_tensor_normal1": shape_tensor_normals_1,
        "shape_tensor_normal2": shape_tensor_normals_2,
        "shape_tensor_eigenvalues": shape_tensor_eigenvalues,
        "R_gramm": R_phis,
        "K_par_eff": K_par_effs,
        "K_perp_eff": K_perp_effs,
    }
    
# Reference data coordinates are rotated -90° around x
Q = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
])
    
def validate_real_case_revs(interpolated_data: dict) -> None:
    """
    Compare interpolated REV data against reference porosity and permeability.
    """

    revData = REVData()

    reference_porosities = revData.get_reference_porosity()
    reference_permeabilities = revData.get_reference_permeabilities()

    values = interpolated_data["porosity"]
    vectors = interpolated_data["wireDirection"]
    shape_tensor_normals_1 = interpolated_data["shape_tensor_normal1"]
    shape_tensor_normals_2 = interpolated_data["shape_tensor_normal2"]
    shape_tensor_eigenvalues = interpolated_data["shape_tensor_eigenvalues"]
    R_phis = interpolated_data["R_gramm"]
    K_par_effs = interpolated_data["K_par_eff"]
    K_perp_effs = interpolated_data["K_perp_eff"]

    

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60 + "\n")

    poro_errors = []
    raw_errors = []
    gramm_errors = []

    for i, (
        porosity,
        vector,
        shape_tensor_normal_1,
        shape_tensor_normal_2,
        eigenvalues,
        R_gramm,
        K_par_eff,
        K_perp_eff,
    ) in enumerate(
        zip(
            values,
            vectors,
            shape_tensor_normals_1,
            shape_tensor_normals_2,
            shape_tensor_eigenvalues,
            R_phis,
            K_par_effs,
            K_perp_effs,
        ),
        start=1,
    ):
        ref_poro = reference_porosities[i]

        poro_err = abs((porosity - ref_poro) / ref_poro)

        R_raw = np.column_stack([
            shape_tensor_normal_1,
            shape_tensor_normal_2,
            vector,
        ])

        err_gramm = permeability_tensor_error(
            Q @ R_gramm,
            reference_permeabilities[i],
        )

        err_raw = permeability_tensor_error(
            Q @ R_raw,
            reference_permeabilities[i],
        )

        poro_errors.append(poro_err)
        raw_errors.append(err_raw)
        gramm_errors.append(err_gramm)

        print(
            f"REV {i:2d} | "
            f"φ err: {poro_err * 100:7.3f}%   "
            f"K raw err: {err_raw * 100:7.3f}%   "
            f"K gramm err: {err_gramm * 100:7.3f}%"
        )

    print("\n" + "-" * 70)
    print(
        f"Mean   | "
        f"φ err: {100 * np.mean(poro_errors):7.3f}%   "
        f"K raw err: {100 * np.mean(raw_errors):7.3f}%   "
        f"K gramm err: {100 * np.mean(gramm_errors):7.3f}%"
    )
    print("-" * 70)
    
def realCaseValidation(N: int = 40) -> None:
    """
    Full real-case validation pipeline.
    """

    vtk_cloud_path = run_real_case_homogenization(N=N)

    interpolated_data = interpolate_real_case_to_revs(
        vtk_cloud_path=vtk_cloud_path,
    )

    validate_real_case_revs(
        interpolated_data=interpolated_data,
    )
    
def realCasePerturbationAnalysis(N: int = 40) -> None:
    revData = REVData()
    reference_permeabilities = revData.get_reference_permeabilities()
    vtk_cloud_path = run_real_case_homogenization(N=N)
    interpolated_data = interpolate_real_case_to_revs(
        vtk_cloud_path=vtk_cloud_path,
    )
    vectors = interpolated_data["wireDirection"]
    shape_tensor_normals_1 = interpolated_data["shape_tensor_normal1"]
    shape_tensor_normals_2 = interpolated_data["shape_tensor_normal2"]
    R_phis = interpolated_data["R_gramm"]
    
    
    all_means_raw = []
    all_stds_raw = []

    all_means_gramm = []
    all_stds_gramm = []
    for i, (
        vector,
        shape_tensor_normal_1,
        shape_tensor_normal_2,
        R_gramm,
    ) in enumerate(
        zip(
            vectors,
            shape_tensor_normals_1,
            shape_tensor_normals_2,
            R_phis,
        ),
        start=1,
    ):

        R_gramm = Q @ R_gramm

        R_raw = Q @ np.column_stack([
            shape_tensor_normal_1,
            shape_tensor_normal_2,
            vector,
        ])
        
        eigenvalues_ref, eigenvectors_ref = np.linalg.eigh(reference_permeabilities[i]) 
        Keff = np.diag(eigenvalues_ref)
        
        [means_gramm_eta, means_raw_eta], [stds_gramm_eta, stds_raw_eta] = perturbation_analysis(Rs=[R_gramm,R_raw], Keff=Keff)


        # store results for this REV
        all_means_raw.append(means_raw_eta)
        all_stds_raw.append(stds_raw_eta)

        all_means_gramm.append(means_gramm_eta)
        all_stds_gramm.append(stds_gramm_eta)
    all_means_raw = np.array(all_means_raw)
    all_stds_raw = np.array(all_stds_raw)

    all_means_gramm = np.array(all_means_gramm)
    all_stds_gramm = np.array(all_stds_gramm)
    
    print("\n" + "=" * 100)
    print("RAW ROTATION")
    print("=" * 100)

    for i in range(12):

        print(
            f"REV {i+1:2d} | "
            f"η=1e-3: {all_means_raw[i,0]:.3e} ± {all_stds_raw[i,0]:.3e} | "
            f"η=1e-2: {all_means_raw[i,1]:.3e} ± {all_stds_raw[i,1]:.3e} | "
            f"η=1e-1: {all_means_raw[i,2]:.3e} ± {all_stds_raw[i,2]:.3e}"
        )

    print("-" * 100)
    print(
        f"Mean   | "
        f"η=1e-3: {np.mean(all_means_raw[:,0]):.3e} ± {np.mean(all_stds_raw[:,0]):.3e} | "
        f"η=1e-2: {np.mean(all_means_raw[:,1]):.3e} ± {np.mean(all_stds_raw[:,1]):.3e} | "
        f"η=1e-1: {np.mean(all_means_raw[:,2]):.3e} ± {np.mean(all_stds_raw[:,2]):.3e}"
    )

    print("\n" + "=" * 100)
    print("GRAMM-SCHMIDT ROTATION")
    print("=" * 100)

    for i in range(12):

        print(
            f"REV {i+1:2d} | "
            f"η=1e-3: {all_means_gramm[i,0]:.3e} ± {all_stds_gramm[i,0]:.3e} | "
            f"η=1e-2: {all_means_gramm[i,1]:.3e} ± {all_stds_gramm[i,1]:.3e} | "
            f"η=1e-1: {all_means_gramm[i,2]:.3e} ± {all_stds_gramm[i,2]:.3e}"
        )

    print("-" * 100)
    print(
        f"Mean   | "
        f"η=1e-3: {np.mean(all_means_gramm[:,0]):.3e} ± {np.mean(all_stds_gramm[:,0]):.3e} | "
        f"η=1e-2: {np.mean(all_means_gramm[:,1]):.3e} ± {np.mean(all_stds_gramm[:,1]):.3e} | "
        f"η=1e-1: {np.mean(all_means_gramm[:,2]):.3e} ± {np.mean(all_stds_gramm[:,2]):.3e}"
    )

    
def revSensitivityAnalysis(N_values=(20, 40, 80)) -> None:
    """
    Sensitivity analysis of REV quantities with respect to grid resolution N.

    Computes, for each REV and each pair of consecutive N values:
      - relative change of porosity
      - relative change of R_phi in Frobenius norm
      - relative change of effective permeability tensor K in Frobenius norm

    Relative change is computed as

        ||A_N2 - A_N1||_F / ||A_N1||_F

    and for porosity as

        |phi_N2 - phi_N1| / |phi_N1|
    """

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

    revData = REVData()
    coords = revData.get_REV_positions(mode="array")

    assign = {0: "x", 1: "y", 2: "z"}

    results = {}

    for N in N_values:
        print(f"\n>>> Running homogenization for N = {N}")

        output_base = os.path.join(
            data_dir,
            f"comparison_analytical_real_case_N{N}"
        )

        vtk_cloud_path = output_base + "_cloud.vtk"

        aneurysm = Aneurysm(coilFilePath, aneuFilePath, vesselFilePath, CLFilePath)
        aneurysm.set_sampling_params(
            scan_count=hc.SCAN_COUNT,
            scan_resolution=hc.SCAN_RESOLUTION,
            sample_point_count=hc.SAMPLE_POINT_COUNT,
            normal_sample_count=hc.NORMAL_SAMPLE_POINT_COUNT,
        )

        aneurysm.generate_grid_and_masks(
            N=N,
            r_REV=0.5,
            padding_distance=0.5,
        )

        homogenizer = Homogenizer(
            aneurysm,
            filtertype="box",
            volume_fraction_wall=0.0,
            wall_boundary_condition="const",
        )

        homogenizer.homogenize_and_output(output_base)

        porosity = interpolate_vtk_field(
            vtk_cloud_path,
            coords,
            "porosity",
            strategy="closest_point",
            radius=0.0,
        )

        R_phis = np.zeros(porosity.shape + (3, 3))

        for i in range(3):
            for j in range(3):
                R_phis[:, i, j] = interpolate_vtk_field(
                    vtk_cloud_path,
                    coords,
                    f"R_{assign[i]}{assign[j]}",
                    strategy="closest_point",
                    radius=0.0,
                )

        K_par_effs = interpolate_vtk_field(
            vtk_cloud_path,
            coords,
            "K_par_eff",
            strategy="closest_point",
            radius=0.0,
        )

        K_perp_effs = interpolate_vtk_field(
            vtk_cloud_path,
            coords,
            "K_perp_eff",
            strategy="closest_point",
            radius=0.0,
        )

        K_tensors = np.zeros(porosity.shape + (3, 3))

        for rev_id in range(len(porosity)):
            K = np.eye(3)
            K[0, 0] = K_perp_effs[rev_id]
            K[1, 1] = K_perp_effs[rev_id]
            K[2, 2] = K_par_effs[rev_id]
            K_tensors[rev_id] = K

        # Store transformed R_phi, matching your validation output
        R_phis_transformed = np.zeros_like(R_phis)
        for rev_id in range(len(porosity)):
            R_phis_transformed[rev_id] = R_phis[rev_id]

        results[N] = {
            "porosity": porosity,
            "R_phi": R_phis_transformed,
            "K": K_tensors,
        }

    print("\n>>> REV SENSITIVITY RESULTS\n")

    sensitivity_stats = {}

    for N_old, N_new in zip(N_values[:-1], N_values[1:]):
        print(f"\n=== Relative change: N = {N_old} -> N = {N_new} ===\n")

        phi_old = results[N_old]["porosity"]
        phi_new = results[N_new]["porosity"]

        R_old = results[N_old]["R_phi"]
        R_new = results[N_new]["R_phi"]

        K_old = results[N_old]["K"]
        K_new = results[N_new]["K"]

        rel_phi_all = []
        rel_R_all = []
        rel_K_all = []

        for rev_id in range(len(phi_old)):
            rel_phi = abs(phi_new[rev_id] - phi_old[rev_id]) / abs(phi_old[rev_id])

            rel_R = (
                np.linalg.norm(R_new[rev_id] - R_old[rev_id], ord="fro")
                / np.linalg.norm(R_old[rev_id], ord="fro")
            )

            rel_K = (
                np.linalg.norm(K_new[rev_id] - K_old[rev_id], ord="fro")
                / np.linalg.norm(K_old[rev_id], ord="fro")
            )

            rel_phi_all.append(100 * rel_phi)
            rel_R_all.append(100 * rel_R)
            rel_K_all.append(100 * rel_K)

            print(
                f"REV {rev_id + 1:2d}: "
                f"porosity = {100 * rel_phi:8.4f}%   "
                f"R_phi = {100 * rel_R:8.4f}%   "
                f"K = {100 * rel_K:8.4f}%"
            )

        rel_phi_all = np.array(rel_phi_all)
        rel_R_all = np.array(rel_R_all)
        rel_K_all = np.array(rel_K_all)

        def summarize(x):
            return {
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "std": float(np.std(x)),
            }

        step_key = f"{N_old}->{N_new}"

        sensitivity_stats[step_key] = {
            "porosity": summarize(rel_phi_all),
            "R_phi": summarize(rel_R_all),
            "K": summarize(rel_K_all),
        }

        print("\nSummary over all REVs:")
        for name, values in sensitivity_stats[step_key].items():
            print(
                f"  {name:8s}: "
                f"mean = {values['mean']:8.4f}%   "
                f"median = {values['median']:8.4f}%   "
                f"std = {values['std']:8.4f}%"
            )

    return sensitivity_stats




def run_blender_script(script_path):
    """
    Run a Blender Python script if Blender is available.
    """
    import shutil
    import subprocess

    expected_folder = "fenicsx-homogenization"
    current_folder = os.path.basename(os.getcwd())

    if current_folder != expected_folder:
        raise RuntimeError(
            f"Wrong directory: currently in '{current_folder}'. "
            f"Please lauch from '{expected_folder}' folder."
        )

    print("Directory check passed.")

    blender = shutil.which("blender")

    if blender is None:
        raise RuntimeError(
            "Blender executable not found in PATH."
        )

    cmd = [
        blender,
        "--background",
        "--python",
        script_path
    ]

    print("Running:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)






