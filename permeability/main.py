import argparse
import itertools
import os

from aneurysm import Aneurysm, test_sampling_grid_properties
from homogenizer import Homogenizer
from validation import realCaseValidation

def run_tests():
    test_sampling_grid_properties()

def run_homogenization(config, geometry_paths, data_subdir):
    """
    Generic homogenization routine shared by realCaseHomogenization()
    and spiralHomogenization().
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Prepare output directory ---
    data_dir = os.path.join(current_dir, "data", data_subdir)
    os.makedirs(data_dir, exist_ok=True)

    # --- Unpack geometry paths ---
    coil_files, aneu, vessel, centerlines = geometry_paths

    # --- Load aneurysm once ---
    aneurysm = Aneurysm(coil_files, aneu, vessel, centerlines)

    aneurysm.set_sampling_params(
        scan_count=config.SCAN_COUNT,
        scan_resolution=config.SCAN_RESOLUTION,
        sample_point_count=config.SAMPLE_POINT_COUNT,
        normal_sample_count=config.NORMAL_SAMPLE_POINT_COUNT
    )

    # Largest REV determines padding
    max_rev = max(config.RADIUS_REV)

    aneurysm.generate_grid_and_masks(
        N=config.SAMPLES_PER_SIDE,
        r_REV=max_rev,
        padding_distance=max_rev
    )

    # --- Sweep all combinations ---
    for radius_REV, filter_type, wall_BC in itertools.product(
        config.RADIUS_REV,
        config.FILTERS,
        config.WALL_BOUNDARY_CONDITIONS
    ):
        aneurysm.set_REV_radius(radius_REV)

        homogenizer = Homogenizer(
            aneurysm,
            filtertype=filter_type,
            wall_boundary_condition=wall_BC[0],
            volume_fraction_wall=wall_BC[1]
        )

        outputname = os.path.join(
            data_dir,
            f"REV_{radius_REV}_Filter_{filter_type}_BC_{wall_BC[2]}"
        )

        homogenizer.homogenize_and_output(outputname)

def realCaseHomogenization():
    import realCaseConfig as config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(current_dir, "..", "geometry", "real_case")

    geometry = (
        [os.path.join(mesh_dir, f) for f in ("coil1.obj", "coil2.obj")],
        os.path.join(mesh_dir, "aneurysm1.obj"),
        os.path.join(mesh_dir, "vessel.obj"),
        [os.path.join(mesh_dir, f) for f in ("spline_1.obj", "spline_2.obj")]
    )

    run_homogenization(config, geometry, data_subdir="real_case")

def artivifialVesselHomogenization():
    import artificialVesselConfig as config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(current_dir, "..", "geometry", "artificial_vessel")

    geometry = (
        [os.path.join(mesh_dir, f) for f in ("coil1.obj", "coil2.obj")],
        os.path.join(mesh_dir, "aneurysm.stl"),
        os.path.join(mesh_dir, "vessel.stl"),
        [os.path.join(mesh_dir, f) for f in ("spline_1.obj", "spline_2.obj")]
    )

    run_homogenization(config, geometry, data_subdir="artificial_vessel")

def spiralHomogenization():
    import homogenizationConfig as config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(current_dir, "..", "geometry", "cylinder_spiral")

    geometry = (
        os.path.join(mesh_dir, "coils_2d.stl"),
        os.path.join(mesh_dir, "domain_2d.stl"),
        os.path.join(mesh_dir, "domain_extended_2d.stl"),
        os.path.join(mesh_dir, "centerlines.obj")
    )

    run_homogenization(config, geometry, data_subdir="cylinder_spiral")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--case",
        choices=["cylinder_spiral", "artificial_vessel", "real_vessel"],
        default="real_vessel",
    )
    args = parser.parse_args()

    if args.case == "cylinder_spiral":
        spiralHomogenization()
    elif args.case == "artificial_vessel":
        artivifialVesselHomogenization()
    elif args.case == "real_vessel":
        realCaseHomogenization()
        # realCaseValidation(40)