# mesh loading
import trimesh

# signed distance computation
import numpy as np
import random
import mesh_to_sdf as mts

class SampleParams():
    def __init__(self):
        self.scan_count = 50,
        self.scan_resolution = 100,
        self.sample_point_count = 1_000_000,
        self.normal_sample_count =11

class Aneurysm():
    def __init__(self, coilFilePath, aneuFilePath: str, vesselFilePath: str, CLFilePath):
        """
        coilFilePath: str or list[str]
        CLFilePath: str or list[str]
        """

        # Normalize inputs: if a single path is given, wrap it in a list
        if isinstance(coilFilePath, str):
            coil_paths = [coilFilePath]
        else:
            coil_paths = list(coilFilePath)

        if isinstance(CLFilePath, str):
            cl_paths = [CLFilePath]
        else:
            cl_paths = list(CLFilePath)

        # Validation: number of centerlines must equal number of coils
        if len(coil_paths) != len(cl_paths):
            raise ValueError(
                f"Number of coils ({len(coil_paths)}) does not match number of centerlines ({len(cl_paths)})."
            )

        # Store data
        self.fileNames = {
            "coil": coil_paths,
            "aneu": aneuFilePath,
            "vessel": vesselFilePath,
            "CL": cl_paths
        }

        self.meshes = {}
        self.sampleParams = SampleParams()

    def load_data(self):
        self.meshes = {}

        # --- Load coils (list of file paths) ---
        coil_meshes = []
        for path in self.fileNames["coil"]:
            coil_meshes.append(trimesh.load(path))
        self.meshes["coil"] = coil_meshes

        # --- Load single meshes ---
        self.meshes["aneu"]   = trimesh.load(self.fileNames["aneu"])
        self.meshes["vessel"] = trimesh.load(self.fileNames["vessel"])

        # --- Load centerlines (list of file paths) ---
        centerlines = []
        for path in self.fileNames["CL"]:
            points, connectivity, curve_ids, num_curves = self.read_centerline_obj(path)
            centerlines.append({
                "points": points,
                "connectivity": connectivity,
                "curve_ids": curve_ids,
                "num_curves": num_curves
            })

        self.centerline = centerlines

    def get_bounding_box(self, mesh_name: str = "aneu"):
        mesh = self.meshes[mesh_name]
        if mesh is None:
            raise ValueError(f"Mesh '{mesh_name}' not loaded!")
        aabb = mesh.bounding_box
        [bounding_box_min, bounding_box_max] = aabb.bounds
        bounding_box_length = aabb.extents
        bounding_box_center = aabb.centroid
        return [bounding_box_min, bounding_box_max, bounding_box_length, bounding_box_center]

    def generate_sample_grid(self, N: int, r_REV: float, padding_distance: float, kappa: float = 1.0, mesh_name: str = "aneu", cell_centered:bool=True):
        """
        Generate a structured 3D Cartesian grid within the mesh bounding box.
        """
        bounding_box_min, bounding_box_max, bounding_box_length, bounding_box_center = self.get_bounding_box(mesh_name)


        self.sample_width = min(bounding_box_length) / (2.0 * N)
        samples_per_dir = np.ceil((  (bounding_box_length) / 2.0 + padding_distance) / self.sample_width).astype(int)
        self.set_REV_radius(r_REV, kappa)

        if not(cell_centered):
            # Create coordinate vectors for each dimension
            x = np.arange(-samples_per_dir[0], samples_per_dir[0] + 1) * self.sample_width + bounding_box_center[0]
            y = np.arange(-samples_per_dir[1], samples_per_dir[1] + 1) * self.sample_width + bounding_box_center[1]
            z = np.arange(-samples_per_dir[2], samples_per_dir[2] + 1) * self.sample_width + bounding_box_center[2]
        else:
            # cell centered version
            x = np.arange(-samples_per_dir[0] , samples_per_dir[0] ) * self.sample_width + bounding_box_center[0] + self.sample_width/2
            y = np.arange(-samples_per_dir[1] , samples_per_dir[1] ) * self.sample_width + bounding_box_center[1] + self.sample_width/2
            z = np.arange(-samples_per_dir[2] , samples_per_dir[2] ) * self.sample_width + bounding_box_center[2] + self.sample_width/2

        # Generate 3D mesh
        self.grid = np.meshgrid(x, y, z, indexing="ij")

        # Flatten and store points + shape
        self.grid_points = np.vstack([self.grid[0].ravel(), self.grid[1].ravel(), self.grid[2].ravel()]).T
        self.grid_shape = (len(x), len(y), len(z))

        print(f"Generated structured grid of shape {self.grid_shape} with {len(self.grid_points)} points.")

    def generate_grid_and_masks(self, N: int = 20, r_REV: float = 1.0, padding_distance: float = None):
        self.load_data()
        if padding_distance == None:
            print("Setting padding distance to REV radius.")
            padding_distance = r_REV
        self.generate_sample_grid(N=N, r_REV=r_REV, padding_distance=padding_distance)
        self.generate_inside_outside_mask()

    def get_flattened_grid(self):
        """
        Return the flattened grid as (N, 3) array of XYZ points.
        """
        if not hasattr(self, "grid"):
            raise RuntimeError("Grid not generated. Call generate_sample_grid() first.")
        return self.flatten_griddata(self.grid)

    def flatten_griddata(self, data):
        # Vector field provided as a tuple/list of 3 components
        if isinstance(data, (tuple, list)):
            if len(data) != 3:
                raise ValueError(f"Expected 3 components, got {len(data)}")
            return np.column_stack([a.ravel(order="F") for a in data])

        # Single NumPy array: scalar or vector field
        if data.ndim == len(self.grid_shape) + 1 and data.shape[-1] == 3:
            return np.column_stack([data[..., i].ravel(order="F") for i in range(3)])
        else:
            return data.ravel(order="F")

    def unflatten_griddata(self, flattened_data):
        if isinstance(flattened_data, (list, tuple)):
            if len(flattened_data) != 3:
                raise ValueError(f"Expected 3 components, got {len(flattened_data)}")
            return np.stack(
                [a.reshape(self.grid_shape, order="F") for a in flattened_data],
                axis=-1
            )
        else:
            return flattened_data.reshape(self.grid_shape, order="F")

    def set_sampling_params(self, scan_count:int=None, scan_resolution:int=None, sample_point_count:int=None, normal_sample_count:int=None):
        if scan_count:
            self.sampleParams.scan_count = scan_count
        if scan_resolution:
            self.sampleParams.scan_resolution = scan_resolution
        if sample_point_count:
            self.sampleParams.sample_point_count = sample_point_count
        if normal_sample_count:
            self.sampleParams.normal_sample_count = normal_sample_count

    def generate_mask_from_mesh(self, mesh, sample_points, deterministic: bool = True):
        if(deterministic):
            np.random.seed(0)
            random.seed(0)
        mask = mts.mesh_to_sdf(mesh, sample_points,
                                        surface_point_method='sample',
                                        sign_method='normal',
                                        bounding_radius=None,
                                        scan_count=self.sampleParams.scan_count,
                                        scan_resolution=self.sampleParams.scan_resolution,
                                        sample_point_count=self.sampleParams.sample_point_count,
                                        normal_sample_count=self.sampleParams.normal_sample_count)<=0
        return self.unflatten_griddata(mask)

    def generate_inside_outside_mask(self, deterministic: bool = True):
        sample_points = self.get_flattened_grid()
        self.masks = dict()

        for key in self.meshes:
            print(f"Generating a bitmask for {key} geometry.")

            # Merge coil meshes if there are multiple
            if key == "coil" and isinstance(self.meshes["coil"], list):
                mesh = trimesh.util.concatenate(self.meshes["coil"])
            else:
                mesh = self.meshes[key]

            self.masks[key] = self.generate_mask_from_mesh(mesh, sample_points, deterministic)

    def generate_inside_outside_mask(self, deterministic: bool = True):
        sample_points = self.get_flattened_grid()
        self.masks = {}

        # Normalize coil(s) into a list so we can iterate consistently
        coil_meshes = (
            self.meshes["coil"]
            if isinstance(self.meshes["coil"], list)
            else [self.meshes["coil"]]
        )

        print(f"Computing masks for {len(coil_meshes)} coil(s).")

        merged_mask = None

        # Iterate through coils and accumulate merged mask
        for i, coil_mesh in enumerate(coil_meshes):
            print(f"  Generating mask for coil {i}")
            mask = self.generate_mask_from_mesh(coil_mesh, sample_points, deterministic)

            # Store individual coil masks in the matching centerline
            if i < len(self.centerline):
                self.centerline[i]["coil_mask"] = mask

            # Accumulate merged coil mask with logical OR
            if merged_mask is None:
                merged_mask = mask.copy()
            else:
                merged_mask |= mask

        # Save merged coil mask under the original key name
        self.masks["coil"] = merged_mask

        # Process aneurysm and vessel as before
        for key in ["aneu", "vessel"]:
            print(f"Generating mask for {key} geometry.")
            mesh = self.meshes[key]
            self.masks[key] = self.generate_mask_from_mesh(mesh, sample_points, deterministic)

    def read_centerline_obj(self, centerline_filename):
        """
        Read a Wavefront .obj centerline file containing one or more polyline curves.

        The OBJ file must contain:
            - Vertex lines of the form:    v x y z
            - Polyline lines of the form:  l i j k ...

        Requirements and behavior:
            - OBJ indices in 'l' lines are 1-based (standard OBJ); these are converted
            internally to 0-based Python indices.
            - A single geometric curve may be defined across multiple 'l' lines.
            - All vertex indices from all 'l' lines are collected and analyzed.
            - Curves are automatically separated wherever the vertex indices are not
            consecutive (i.e., a break in topology).
            - The function reconstructs clean, continuous polylines (curves) from the
            raw OBJ connectivity.

        Returns
        -------
        points : (N, 3) ndarray
            Array of vertex coordinates.
        connectivity : (N, 2) ndarray
            For each vertex: [next_vertex_index, previous_vertex_index].
            A value of -1 indicates no neighbor in that direction.
        curve_ids : (N,) ndarray
            Integer identifier for the curve to which each vertex belongs
            (0, 1, 2, ...).
        num_curves : int
            Total number of distinct polyline curves detected.
        """

        # --- Validate input ---
        if not centerline_filename.lower().endswith('.obj'):
            raise ValueError("Expected a .obj file extension.")

        vertex_positions = []
        polyline_vertex_indices = []  # each entry = list of vertex indices for one curve

        # --- Parse OBJ file ---
        with open(centerline_filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    # Vertex line: 'v x y z'
                    parts = line.strip().split()
                    vertex_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('l '):
                    # Polyline definition: 'l i j k ...'
                    # Convert from 1-based OBJ indices to 0-based Python indices
                    vertex_ids = [int(idx) - 1 for idx in line.strip().split()[1:]]
                    polyline_vertex_indices.append(vertex_ids)

        # --- Convert vertices to array ---
        points = np.array(vertex_positions, dtype=float)
        num_points = points.shape[0]

        # --- Initialize connectivity and curve mapping ---
        connectivity = -1 * np.ones((num_points, 2), dtype=int)  # [next, prev]
        curve_ids = -1 * np.ones(num_points, dtype=int)

        # --- Assign connectivity per curve ---
        for curve_index, vertex_ids in enumerate(polyline_vertex_indices):
            for local_idx, vertex_id in enumerate(vertex_ids):
                curve_ids[vertex_id] = curve_index

                # Forward connection
                if local_idx < len(vertex_ids) - 1:
                    connectivity[vertex_id, 0] = vertex_ids[local_idx + 1]
                # Backward connection
                if local_idx > 0:
                    connectivity[vertex_id, 1] = vertex_ids[local_idx - 1]

        num_curves = len(polyline_vertex_indices)

        return points, connectivity, curve_ids, num_curves

    def get_domain_mask(self):
        return self.masks["aneu"]

    def get_obstacle_mask(self):
        return self.masks["coil"]

    def get_full_domain_mask(self):
        return self.masks["vessel"]

    def get_discrete_REV_radius(self) -> int:
        return self.r_REV_discrete_scaled

    def get_REV_radius(self) -> float:
        return self.r_REV_scaled

    def get_centerline_dict(self) -> dict:
        return self.centerline

    def set_REV_radius(self, r_REV: float, kappa:float=1.0):
        self.r_REV_scaled = r_REV*kappa
        self.r_REV_discrete_scaled = int(np.round(self.r_REV_scaled / self.sample_width))


# ===========================================================
# --- TESTS ----------
# ===========================================================

def test_sampling_grid_properties():
    """
    Test that the generated sample grid:
      1. Is centered at the mesh centroid
      2. Is elongated by r_REV on each sides
    """
    import math
    # --- Create a simple cubic test mesh ---
    mesh_lengths = [10.0, 8.0, 6.0]

    mesh = trimesh.creation.box(extents=mesh_lengths)
    # generate aneurysm object with dummy paths
    aneurysm = Aneurysm("coil.obj", "aneu.obj", "vessel.obj", "CL.obj")
    aneurysm.meshes["aneu"] = mesh  # directly assign test mesh

    N = 10
    r_REV = 1.0
    aneurysm.generate_sample_grid(N=N, r_REV=r_REV, mesh_name="aneu")

    expected_sample_width = min(mesh_lengths)/(2*N)

    # --- Extract reference bounding box and generated grid ---
    bb_min, bb_max, bb_len, bb_center = aneurysm.get_bounding_box("aneu")
    X, Y, Z = aneurysm.grid
    sample_width = aneurysm.sample_width
    grid_min = np.array([X.min(), Y.min(), Z.min()])
    grid_max = np.array([X.max(), Y.max(), Z.max()])
    grid_center = (grid_min + grid_max) / 2.0
    grid_extent = grid_max - grid_min

    tol = 1e-8

    assert np.allclose(grid_center, bb_center, atol=tol), \
        f"Grid center {grid_center} ≠ mesh center {bb_center}"

    # --- Check elongation by r_REV per side ---
    expected_extent = [math.ceil( (mesh_lengths[i]/2.0 + r_REV)/expected_sample_width)*expected_sample_width*2.0 for i in range(3)]
    assert np.allclose(grid_extent, expected_extent, atol=aneurysm.sample_width+sample_width*2), \
        f"Grid extent {grid_extent} ≠ expected {expected_extent}"

    print("  Sampling grid test passed:")
    print(f"   Center OK ({grid_center})")
    print(f"   Extent OK ({grid_extent})")
