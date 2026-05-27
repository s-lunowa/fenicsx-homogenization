import numpy as np
from scipy import ndimage, interpolate, spatial

import utils

class Homogenizer():
    def __init__(self, domain, filtertype: str, volume_fraction_wall: float=1.0, wall_boundary_condition: str="const"):
        self.domain = domain
        self.homogenization_domain = domain.get_domain_mask()
        self.homogenization_obstacle = domain.get_obstacle_mask()
        self.homogenization_full_domain = domain.get_full_domain_mask()
        self.discrete_scaled_REV_radius = domain.get_discrete_REV_radius()
        self.centerline_dict = domain.get_centerline_dict()
        self.volume_fraction_wall = volume_fraction_wall
        self.wall_boundary_condition = wall_boundary_condition
        self.filtertype = filtertype
        # restrict obstace to homogenization domain
        self.homogenization_obstacle[~self.homogenization_domain] = False

    def homogenize_and_output(self, outputname: str):
        self._construct_boundary_conditions()
        self._homogenize_volume_fraction()
        self._homogenize_obstacle_centerline()
        self._generate_permeability()
        self._rotation_from_tangent()
        self._generate_output(outputname)

    def _generate_permeability(self, effectiveRadius: float = 0.1778/1000):
        """
        Compute analytical permeabilities (Eq. 9–10) and construct K_phi (Eq. 11).
        Only the symmetric upper triangular entries are returned.

        Inputs
        ------
        rho : porosity (scalar OR field)
        wireDir : 3-vector direction field (normalized)
        R : effective fibre radius
        """
        rho = 1-self.porosity.flatten(order="F")
        d   = self.homogenized_tangent_flat  # should be normalized 3-vector
        #d   = d / np.linalg.norm(d)

        R = float(effectiveRadius)

        # Clip to avoid log(0)
        rho = np.clip(rho, 1e-6, 1 - 1e-6)

        # ===== Equation (9): statically continuous ("p") =====
        K_perp_p = -(R**2 / (8.0 * rho)) * ( np.log(rho) + (1 - rho**2)/(1 + rho**2) )
        K_par_p  = -(R**2 / (4.0 * rho))       * ( np.log(rho) + ((1 - rho)*(3 - rho))/2.0 )

        # ===== Equation (10): kinematically continuous ("v") =====
        K_perp_v = -(R**2 / (8.0 * rho)) * ( np.log(rho) + (2*(1 - rho))/(1 + rho) )
        K_par_v  = -(R**2 / (4.0 * rho))       * ( np.log(rho) + (2*(1 - rho))/(1 + rho) )

        # =====================================================
        # Eq. (11): weighted effective permeability K_phi
        # longitudinal and transverse effective coefficients
        # =====================================================

        K_par_eff  = (3*K_par_p + K_par_v) / 4.0     # scalar
        K_perp_eff = (K_perp_p + K_perp_v) / 2.0     # scalar

        # Outer product d ⊗ d
        ddT = np.array([d[:,0]*d[:,0],
                        d[:,0]*d[:,1],
                        d[:,0]*d[:,2],
                        d[:,1]*d[:,1],
                        d[:,1]*d[:,2],
                        d[:,2]*d[:,2]])

        # Identity
        I = np.array([np.ones_like(d[:,0]),
                      np.zeros_like(d[:,0]),
                      np.zeros_like(d[:,0]),
                      np.ones_like(d[:,0]),
                      np.zeros_like(d[:,0]),
                      np.ones_like(d[:,0])])
        K_par_eff_tens = np.array([K_par_eff]*6)
        K_perp_eff_tens = np.array([K_perp_eff]*6)

        self.K_par_eff = K_par_eff
        self.K_perp_eff = K_perp_eff
        # Full K_phi tensor
        self.K_phi = K_par_eff_tens * ddT + K_perp_eff_tens * (I - ddT)

    def _rotation_from_tangent(self):
        """
        Construct a 3×3 orthonormal rotation matrix R such that:

            R[:,-1] = e1 = normalized(self.homogenized_tangent)

        Remaining columns e2, e3 are built via a stable Gram–Schmidt process.

        Returns
        -------
        R : (N, 3, 3) numpy.ndarray (orthonormal) where N is number of tangent vectors
            or (3, 3) numpy.ndarray if single tangent vector
        """
        # Extract tangent vector(s)
        d = self.homogenized_tangent_flat

        # Handle both single vector and multiple vectors
        if d.ndim == 1:
            d = d.reshape(1, -1)
            return_single = True
        else:
            return_single = False

        n_vectors = d.shape[0]

        # Normalize e1 (avoid division by zero)
        norm_d = np.linalg.norm(d, axis=1, keepdims=True)
        if np.any(norm_d < 1e-14):
            raise ValueError("Tangent vector is zero; cannot build rotation matrix.")

        e1 = d / norm_d

        # Initialize arrays for e2 and e3
        e2 = np.zeros_like(d)
        e3 = np.zeros_like(d)

        # Build orthonormal basis for each tangent vector
        for i in range(n_vectors):
            e1_i = e1[i]

            # Pick auxiliary vector not parallel to e1_i
            # Use the basis vector that is most orthogonal to e1_i
            abs_e1 = np.abs(e1_i)
            min_idx = np.argmin(abs_e1)

            if min_idx == 0:
                a = np.array([1.0, 0.0, 0.0])
            elif min_idx == 1:
                a = np.array([0.0, 1.0, 0.0])
            else:
                a = np.array([0.0, 0.0, 1.0])

            # Gram–Schmidt: e2 = normalized(a - (a·e1)e1)
            a_dot_e1 = np.dot(a, e1_i)
            v2 = a - a_dot_e1 * e1_i
            norm_v2 = np.linalg.norm(v2)

            if norm_v2 < 1e-14:
                # Fallback: use a different auxiliary vector
                # Try the next smallest component
                sorted_indices = np.argsort(abs_e1)
                for idx in sorted_indices[1:]:
                    if idx == 0:
                        a_alt = np.array([1.0, 0.0, 0.0])
                    elif idx == 1:
                        a_alt = np.array([0.0, 1.0, 0.0])
                    else:
                        a_alt = np.array([0.0, 0.0, 1.0])

                    a_dot_e1_alt = np.dot(a_alt, e1_i)
                    v2_alt = a_alt - a_dot_e1_alt * e1_i
                    norm_v2_alt = np.linalg.norm(v2_alt)

                    if norm_v2_alt >= 1e-14:
                        v2 = v2_alt
                        norm_v2 = norm_v2_alt
                        break
                else:
                    raise ValueError("Failed to construct orthogonal vector (numerical issue).")

            e2[i] = v2 / norm_v2

            # e3 = e1 × e2
            e3[i] = np.cross(e2[i], e1_i)#np.cross(e1_i, e2[i])

        # Build rotation matrices
        if return_single:
            # Single (3, 3) rotation matrix
            R = np.column_stack([e2[0], e3[0], e1[0]])
        else:
            # Multiple (N, 3, 3) rotation matrices
            R = np.stack([e2, e3, e1], axis=2)

        self.R = R

    def _construct_boundary_conditions(self):
        self.volume_fraction_not_averaged = np.zeros_like(self.homogenization_full_domain).astype("float")
        if self.wall_boundary_condition == "const":
            phi_wall = self.volume_fraction_wall
        elif self.wall_boundary_condition == "avg":
            domain_points = np.count_nonzero(self.homogenization_domain)
            obstacle_points = np.count_nonzero(self.homogenization_obstacle)
            if obstacle_points == 0:
                raise ValueError("Domain empty. Cannot calculate average.")
            phi_wall = obstacle_points / domain_points
        elif self.wall_boundary_condition == "reflection":
            phi_wall = 1.0 
        self.phi_wall = phi_wall
        self.volume_fraction_not_averaged[~self.homogenization_full_domain] = phi_wall
        self.volume_fraction_not_averaged[self.homogenization_obstacle] = 1.0

    def _homogenize_volume_fraction(self):
        if self.filtertype=="gaussian":
            filter = ndimage.gaussian_filter
            sigma = self.discrete_scaled_REV_radius/2 # sigma is in neighbour units not distance
            print(f"Applying gauss averaging with sigma of {sigma}.")
            kwargs = {"sigma": sigma}
        else:
            filter = ndimage.uniform_filter
            size = 2 * self.discrete_scaled_REV_radius + 1
            print(f"Applying box averaging with descrete REV diameter of {size}.")
            kwargs = {"size": size}
        if self.wall_boundary_condition=="reflection":
            print("Performing homogenization with reflection.")
            eps = 1e-12
            full_domain_local_mean = filter(self.homogenization_full_domain.astype('float'), mode='constant', cval=0, **kwargs)
            self.volume_fraction_averaged = filter( (self.volume_fraction_not_averaged*self.homogenization_full_domain).astype('float'), mode='constant', cval=0, **kwargs)
            self.volume_fraction_averaged[self.homogenization_domain] /= (full_domain_local_mean[self.homogenization_domain] + eps)
        else:
            print("Performing homogenization without reflection.")
            self.volume_fraction_averaged = filter(self.volume_fraction_not_averaged.astype('float'), mode='constant', cval=self.phi_wall, **kwargs)
        self.porosity = 1.0 - self.volume_fraction_averaged

    def _compute_tangent_field_from_centerlines(self, samples_per_seg=10, smooth=0.0, chunk=50000, write_output=True):
        """
        Compute a tangent vector field from the stored centerlines
        and project it onto the voxel grid (inside the coil mask).

        Uses Aneurysm.get_flattened_grid() for coordinate consistency.
        """
        print("Computing tangent field from centerlines...")

        # --- Extract grid info ---
        grid_points = self.domain.get_flattened_grid()  # (N, 3)
        nPts = grid_points.shape[0]

        # --- Prepare outputs (flat arrays for efficiency) ---
        tan_flat = np.zeros((nPts, 3), dtype=float)
        has_tangent_flat = np.zeros(nPts, dtype=bool)

        # --- Centerline data ---
        centerline_dicts = self.domain.get_centerline_dict()
        for centerline_dict in centerline_dicts:

            # --- Coil region mask (same shape as grid) ---
            coil_mask = centerline_dict["coil_mask"]
            if not np.any(coil_mask):
                print("Coil mask is empty — no tangents to compute.")
                zeros = np.zeros_like(coil_mask)
                return zeros, zeros, zeros, zeros

            # --- Flatten mask for indexing ---
            coil_mask_flat = coil_mask.flatten(order="F")
            inside_idx = np.where(coil_mask_flat)[0]
            if inside_idx.size == 0:
                print("No active voxels in coil mask.")
                zeros = np.zeros_like(coil_mask)
                return zeros, zeros, zeros, zeros

            centerline_points = centerline_dict["points"]
            centerline_connectivity = centerline_dict["curve_ids"]
            curve_ids = np.unique(centerline_connectivity)

            all_points = []
            all_tangents = []

            # --- Fit splines for each curve ---
            for curve_id in curve_ids:
                pts = centerline_points[centerline_connectivity == curve_id]
                if pts.shape[0] < 4:
                    continue

                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
                try:
                    tck, u = interpolate.splprep([x, y, z], s=smooth, k=min(3, len(pts) - 1))
                except Exception as e:
                    print(f"Skipping curve {curve_id}: spline fit failed ({e})")
                    continue

                num_samples = max(2, (len(pts) - 1) * samples_per_seg)
                u_dense = np.linspace(0, 1, num_samples)
                xd, yd, zd = interpolate.splev(u_dense, tck)
                dxu, dyu, dzu = interpolate.splev(u_dense, tck, der=1)

                P_dense = np.column_stack([xd, yd, zd])
                T_dense = np.column_stack([dxu, dyu, dzu])
                T_unit = T_dense / (np.linalg.norm(T_dense, axis=1, keepdims=True) + 1e-12)

                all_points.append(P_dense)
                all_tangents.append(T_unit)

            if not all_points:
                print("No valid centerlines found — tangent field empty.")
                zeros = np.zeros_like(coil_mask)
                return zeros, zeros, zeros, zeros

            # --- Combine all curve data ---
            all_points = np.vstack(all_points)
            all_tangents = np.vstack(all_tangents)

            # --- Build KDTree for tangent projection ---
            tree = spatial.cKDTree(all_points)

            # --- Points inside coil ---
            query_points = grid_points[inside_idx, :]

            print(f" Querying {len(query_points)} coil voxels for nearest centerline points...")

            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = lambda x: x
            for start in tqdm(range(0, len(query_points), chunk)):
                stop = min(start + chunk, len(query_points))
                _, idx = tree.query(query_points[start:stop], k=1)
                t_vec = all_tangents[idx]

                tan_flat[inside_idx[start:stop]] = t_vec
                has_tangent_flat[inside_idx[start:stop]] = True

        print(" Tangent field computed successfully.")

        self.has_tangent_flat = has_tangent_flat
        self.tan_flat = tan_flat
        # --- Reshape to 3D grids ---
        self.has_tangent = self.domain.unflatten_griddata(has_tangent_flat)
        self.tanX = self.domain.unflatten_griddata(tan_flat[:, 0])
        self.tanY = self.domain.unflatten_griddata(tan_flat[:, 1])
        self.tanZ = self.domain.unflatten_griddata(tan_flat[:, 2])
        return self.tanX , self.tanY, self.tanZ, self.has_tangent

    def _tangent_shape_tensor(self, tanX, tanY, tanZ, hasTangent):
        """
        Compute a direction-invariant tensor field from tangent vectors.

        Each voxel's tensor is: S = T * T^T, where T is the normalized tangent vector.

        Parameters
        ----------
        tanX, tanY, tanZ : float arrays (nx, ny, nz)
            Tangent vector components.
        hasTangent : bool array (nx, ny, nz)
            Mask of voxels where tangent is defined.

        Returns
        -------
        Sxx, Sxy, Sxz, Syy, Syz, Szz : float arrays (nx, ny, nz)
            Components of the symmetric tensor field.
        """
        # Initialize tensor component arrays
        Sxx = np.zeros_like(tanX)
        Sxy = np.zeros_like(tanX)
        Sxz = np.zeros_like(tanX)
        Syy = np.zeros_like(tanX)
        Syz = np.zeros_like(tanX)
        Szz = np.zeros_like(tanX)

        # Normalize tangents (just in case)
        mag = np.sqrt(tanX**2 + tanY**2 + tanZ**2) + 1e-12
        Tx = np.where(hasTangent, tanX / mag, 0)
        Ty = np.where(hasTangent, tanY / mag, 0)
        Tz = np.where(hasTangent, tanZ / mag, 0)

        # Compute symmetric tensor components
        Sxx[hasTangent] = Tx[hasTangent] * Tx[hasTangent]
        Syy[hasTangent] = Ty[hasTangent] * Ty[hasTangent]
        Szz[hasTangent] = Tz[hasTangent] * Tz[hasTangent]
        Sxy[hasTangent] = Tx[hasTangent] * Ty[hasTangent]
        Sxz[hasTangent] = Tx[hasTangent] * Tz[hasTangent]
        Syz[hasTangent] = Ty[hasTangent] * Tz[hasTangent]

        return Sxx, Sxy, Sxz, Syy, Syz, Szz

    def _smooth_tangent_tensor_field(self, Sxx, Sxy, Sxz, Syy, Syz, Szz, rREV=1, mode='constant'):
        """
        Smooth the tangent tensor field using a Gaussian (or uniform) filter.

        Parameters
        ----------
        Sxx, Sxy, Sxz, Syy, Syz, Szz : float arrays
            Components of the tangent tensor field.
        sigma : float
            Standard deviation of Gaussian kernel (in voxels).
        mode : str
            Boundary condition for scipy.ndimage filters ('reflect', 'nearest', 'constant', etc.).

        Returns
        -------
        TsmX, TsmY, TsmZ : float arrays
            Smoothed, direction-invariant tangent vectors.
        """
        sigma = rREV/2
        # --- Smooth each tensor component ---
        Sxx_s = ndimage.gaussian_filter(Sxx, sigma=sigma, mode=mode, cval=0.0)
        Sxy_s = ndimage.gaussian_filter(Sxy, sigma=sigma, mode=mode, cval=0.0)
        Sxz_s = ndimage.gaussian_filter(Sxz, sigma=sigma, mode=mode, cval=0.0)
        Syy_s = ndimage.gaussian_filter(Syy, sigma=sigma, mode=mode, cval=0.0)
        Syz_s = ndimage.gaussian_filter(Syz, sigma=sigma, mode=mode, cval=0.0)
        Szz_s = ndimage.gaussian_filter(Szz, sigma=sigma, mode=mode, cval=0.0)

        # --- Reconstruct local tensors ---
        nx, ny, nz = Sxx.shape
        TsmX = np.zeros_like(Sxx)
        TsmY = np.zeros_like(Sxx)
        TsmZ = np.zeros_like(Sxx)

        R_full = np.array([[np.zeros_like(Sxx), np.zeros_like(Sxx), np.zeros_like(Sxx)],
                           [np.zeros_like(Sxx), np.zeros_like(Sxx), np.zeros_like(Sxx)],
                           [np.zeros_like(Sxx), np.zeros_like(Sxx), np.zeros_like(Sxx)]])

        V_eig = np.array([np.zeros_like(Sxx), np.zeros_like(Sxx), np.zeros_like(Sxx)])

        # --- Extract dominant eigenvector for each voxel ---
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    S = np.array([[Sxx_s[i,j,k], Sxy_s[i,j,k], Sxz_s[i,j,k]],
                                [Sxy_s[i,j,k], Syy_s[i,j,k], Syz_s[i,j,k]],
                                [Sxz_s[i,j,k], Syz_s[i,j,k], Szz_s[i,j,k]]])
                    vals, vecs = np.linalg.eigh(S)
                    # largest eigenvector gives dominant orientation
                    T = vecs[:, np.argmax(vals)]
                    R_full[:,:,i,j,k] = vecs
                    V_eig[:,i,j,k] = vals
                    TsmX[i,j,k], TsmY[i,j,k], TsmZ[i,j,k] = T

        return TsmX, TsmY, TsmZ, V_eig, R_full

    def _homogenize_obstacle_centerline(self):
        tanX, tanY, tanZ, hasTangent = self._compute_tangent_field_from_centerlines()
        Sxx, Sxy, Sxz, Syy, Syz, Szz = self._tangent_shape_tensor(tanX, tanY, tanZ, hasTangent)
        TsmX, TsmY, TsmZ, V_eig, R_full = self._smooth_tangent_tensor_field(Sxx, Sxy, Sxz, Syy, Syz, Szz, rREV=self.discrete_scaled_REV_radius)
        self.homogenized_normals1_flat = self.domain.flatten_griddata(np.stack([R_full[0,0], R_full[1,0], R_full[2,0]], axis=-1))
        self.homogenized_normals2_flat = self.domain.flatten_griddata(np.stack([R_full[0,1], R_full[1,1], R_full[2,1]], axis=-1))
        self.V_eig = self.domain.flatten_griddata(np.stack([V_eig[0], V_eig[1], V_eig[2]], axis=-1))
        self.homogenized_tangent_flat = self.domain.flatten_griddata(np.stack([TsmX, TsmY, TsmZ], axis=-1))  # shape (N, 3)

    def _generate_output(self, output_filename: str):
        fields = dict()
        fields["porosity"] = self.porosity
        fields["volume_fraction_not_averaged"] = self.volume_fraction_not_averaged
        fields["wireDirection_raw"] = self.tan_flat
        fields["shape_tensor_normal1"] = self.homogenized_normals1_flat
        fields["shape_tensor_normal2"] = self.homogenized_normals2_flat
        fields["shape_tensor_eigenvalues"] = self.V_eig
        fields["wireDirection"] = self.homogenized_tangent_flat
        fields["R_xx"] = self.R[:,0,0]
        fields["R_xy"] = self.R[:,0,1]
        fields["R_xz"] = self.R[:,0,2]
        fields["R_yx"] = self.R[:,1,0]
        fields["R_yy"] = self.R[:,1,1]
        fields["R_yz"] = self.R[:,1,2]
        fields["R_zx"] = self.R[:,2,0]
        fields["R_zy"] = self.R[:,2,1]
        fields["R_zz"] = self.R[:,2,2]
        fields["K_par_eff"] = self.K_par_eff
        fields["K_perp_eff"] = self.K_perp_eff
        for key in self.domain.masks:
            fields[key] = self.domain.masks[key]
        mask_aneurysm = self.domain.flatten_griddata(self.domain.masks["aneu"])

        utils.write_grid_vtk_cellcentered(self.domain.grid, self.domain.grid_shape, self.domain.sample_width, output_filename, **fields)
        utils.write_grid_vtk_pointcloud(self.domain.get_flattened_grid(), self.domain.grid_shape, output_filename+"_cloud", mask_aneurysm, **fields)
        utils.write_grid_hdf5_pointcloud(self.domain.get_flattened_grid(), self.domain.grid_shape, output_filename+"_cloud", mask_aneurysm, **fields)
