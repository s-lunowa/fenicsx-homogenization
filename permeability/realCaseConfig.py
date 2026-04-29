
# Grid and sampling by SDF generation
SAMPLES_PER_SIDE = 40
SCAN_COUNT = 100              # Number of virtual scans (camera viewpoints) around the mesh.
                            # Higher = better surface coverage, slower runtime.

SCAN_RESOLUTION = 100        # Pixel resolution of each virtual depth scan.
                            # Higher = finer surface detail, slower and more memory-intensive.

SAMPLE_POINT_COUNT = 1_000_000  # Number of points to sample from the mesh surface (used when
                                # surface_point_method='sample'). Ignored when using 'scan'.
                                # Higher = denser surface representation, more accurate distances.

NORMAL_SAMPLE_POINT_COUNT = 11   # Number of nearby surface points checked when using
                                 # sign_method='normal'. The sign is chosen by majority vote.
                                 # Higher = slightly smoother signs, but slower.


# Averaging parameters
RADIUS_REV = [0.25]
FILTERS = ["gaussian"]
WALL_BOUNDARY_CONDITIONS = [["const", 1.0, "wall"]]#,
                            #["const", 0.0, "fluid"],
                            #["avg", None, "avg"],
                            #["reflection", None, "reflection"]]