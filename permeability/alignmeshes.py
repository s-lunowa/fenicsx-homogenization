import pymeshlab as ml
import numpy as np

# ----------------------
# Load meshes correctly
# ----------------------
ms = ml.MeshSet()
ms.load_new_mesh("/home/fabian/Desktop/align/coils_frame_fill.obj")    # mesh 0
ms.load_new_mesh("/home/fabian/Desktop/align/splines_merged_aligned_trimmed.obj")   # mesh 1


# ----------------------
# Extract vertex arrays
# ----------------------
V_ref = np.array(ms.mesh(0).vertex_matrix(), float)
V_mov = np.array(ms.mesh(1).vertex_matrix(), float)

# ----------------------
# PCA function
# ----------------------
def pca(X):
    C = X.mean(axis=0)
    Xc = X - C
    eigvals, eigvecs = np.linalg.eigh(np.cov(Xc.T))
    idx = np.argsort(eigvals)[::-1]
    R = eigvecs[:, idx]
    return R, C  # rotation matrix, centroid

# PCA for both meshes
R_ref, C_ref = pca(V_ref)
R_mov, C_mov = pca(V_mov)

# ----------------------
# Rotation matrix aligning mov → ref
# ----------------------
R = R_ref @ R_mov.T

# Build full 4×4 matrix
M_rot = np.eye(4)
M_rot[:3, :3] = R

# ----------------------
# Apply rotation
# ----------------------
ms.set_current_mesh(1)
ms.apply_filter("set_matrix",
    transformmatrix=M_rot,
    compose=False,
    freeze=True,
    alllayers=False
)

# ----------------------
# Recompute centroid after rotation
# ----------------------
V_mov_rot = np.array(ms.mesh(1).vertex_matrix(), float)
C_mov_rot = V_mov_rot.mean(axis=0)

# Translation needed: t = C_ref - C_mov_rot
t = C_ref - C_mov_rot

# Build translation matrix
M_trans = np.eye(4)
M_trans[:3, 3] = t

# Apply translation
ms.apply_filter("set_matrix",
    transformmatrix=M_trans,
    compose=False,
    freeze=True,
    alllayers=False
)

# Save result
ms.save_current_mesh("/home/fabian/Desktop/align/aligned_splines.obj")
print("PCA alignment complete.")


# =====================================================================
# STEP 2 — Functions for safe spline OBJ transformation
# =====================================================================

def load_obj_vertices(path):
    """Load vertex positions and preserve all non-vertex lines."""
    verts = []
    other = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            else:
                other.append(line)
    return np.array(verts, float), other


def save_obj(path, V, other_lines):
    """Write back OBJ with transformed vertices + original curve data."""
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for line in other_lines:
            f.write(line)


def transform_spline_vertices(V, M_rot, t):
    """Apply rotation+translation to Nx3 vertex array."""
    R = M_rot[:3, :3]   # ensure pure 3x3 rotation
    V_rot = V @ R.T
    V_final = V_rot + t
    return V_final


def transform_obj_spline(path_in, path_out, M_rot, t):
    """Load spline OBJ → transform vertices → save new OBJ."""
    V, other = load_obj_vertices(path_in)
    V2 = transform_spline_vertices(V, M_rot, t)
    save_obj(path_out, V2, other)
    print("Transformed spline:", path_out)



# =====================================================================
# STEP 3 — Apply transform to spline OBJ files
# =====================================================================

# extra_meshes = [
#     "/home/fabian/Desktop/align/spline_1.obj",
#     "/home/fabian/Desktop/align/spline_2.obj"
# ]

# for path in extra_meshes:
#     out = path.replace(".obj", "_aligned.obj")
#     transform_obj_spline(path, out, M_rot, t)

# print("All splines aligned successfully.")

# import mesh_to_sdf 
# def trim_spline_with_sdf(V, coil_mesh):
#     """
#     Uses mesh_to_sdf() with custom scan/sample parameters
#     to classify spline points as inside/outside.
#     Points with SDF < 0 are inside the coil-frame mesh.
#     """
#     sdf = mesh_to_sdf.mesh_to_sdf(
#         mesh=coil_mesh,
#         query_points=V,
#         surface_point_method='scan',
#         sign_method='normal',
#         scan_count=50,
#         scan_resolution=100,
#         sample_point_count=1_000_000,
#         normal_sample_count=11
#     )

#     mask = sdf < 0      # inside → negative SDF
#     return V[mask], mask.sum()


# import trimesh
# coil_mesh = trimesh.load_mesh("/home/fabian/Desktop/align/coils_frame_fill.obj")
# for path in extra_meshes:

#     aligned_path = path.replace(".obj", "_aligned.obj")

#     # 1 — Load ALIGNED spline vertices using trimesh
#     spline_mesh = trimesh.load_mesh(aligned_path, process=False)
#     V2 = spline_mesh.vertices.copy()

#     # 2 — Trim using SDF
#     V_trimmed, n_kept = trim_spline_with_sdf(V2, coil_mesh)

#     # 3 — Load original OBJ to recover curve definitions
#     _, other = load_obj_vertices(path)  # only curve lines

#     # 4 — Save trimmed aligned spline
#     out = path.replace(".obj", "_aligned_trimmed.obj")
#     save_obj(out, V_trimmed, other)

#     print(f"{path}: kept {n_kept}/{len(V2)} points → saved {out}")