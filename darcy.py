# Mixed RT-DG elements for Darcy flow
# (requires dolfinx 0.9.0 or 0.10.0)
#
# Author: S.B. Lunowa

import argparse
import numpy as np
import os

from dolfinx import default_real_type, fem, io, mesh
from dolfinx import __version__ as DX_VERSION
if DX_VERSION == "0.9.0":
    from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
else:
    from dolfinx.fem.petsc import LinearProblem
import ufl

from mpi4py import MPI
from petsc4py import PETSc

def root_print(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

# Parameters and mesh
parser = argparse.ArgumentParser()
# Case names: K = k(phi) Id, K = K(phi), K = K_homogenized
parser.add_argument("-c", "--case", choices=["isotropic", "anisotropic", "anisotropic_p", "anisotropic_v", "anisotropic_avg", "homogenized"], default="isotropic")
parser.add_argument("-e", "--epsilon", type=float, default=0.25, help="Averaging scaling epsilon")
parser.add_argument("-m", "--monolithic", action=argparse.BooleanOptionalAction, default=False, help="Use monolithic solver instead of block solver")
parser.add_argument("-n", "--nx", type=int, default=32, help="Number of vertices per dim: dx = 1/nx")
parser.add_argument("-o", "--order", type=int, default=1, help="FEM order")
args = parser.parse_args()
root_print(f"Options used: {args}", flush=True)

radius = 0.25;  # coil radius
mu = 1e-3 # dynamic viscosity
rel_tol = 1e-14

msh = mesh.create_unit_cube(MPI.COMM_WORLD, args.nx, args.nx, args.nx)
dim = msh.geometry.dim

# derived parameters
fillRatio = np.pi * np.sqrt(3.0) * radius **2 # volume of cylinder in unit cell
porosity = 1.0 - fillRatio
if args.case == "isotropic":
    K = (2.0 * radius * args.epsilon)**2 * porosity**3 / (150 * fillRatio**2) # isotropic permeability
elif args.case == "homogenized":
    K = fem.Constant(msh, 1e-3 * args.epsilon**2 * np.array([[6.24, 1.59, 1.59], [1.59, 6.24, 1.59], [1.59, 1.59, 6.24]], dtype=default_real_type))
else:
    k_pp = -(radius * args.epsilon)**2 / (4 * fillRatio) * (np.log(fillRatio) + 0.5 * (1 - fillRatio) * (3 - fillRatio))
    k_po = -(radius * args.epsilon)**2 / (8 * fillRatio) * (np.log(fillRatio) + (1 - fillRatio**2) / (1 + fillRatio**2))
    k_vp = -(radius * args.epsilon)**2 / (4 * fillRatio) * (np.log(fillRatio) + 2 * (1 - fillRatio) / (1 + fillRatio))
    k_vo = -(radius * args.epsilon)**2 / (8 * fillRatio) * (np.log(fillRatio) + 2 * (1 - fillRatio) / (1 + fillRatio))
    # K = V D V^T
    v1 = 1/np.sqrt(3)
    d = np.array([v1, v1, v1], dtype=default_real_type)
    if args.case == "anisotropic":
        k_parallel = 0.75 * k_pp + 0.25 * k_vp
        k_orthogonal = 0.5 * (k_po + k_vo)
    elif args.case == "anisotropic_p":
        k_parallel = k_pp
        k_orthogonal = k_po
    elif args.case == "anisotropic_v":
        k_parallel = k_vp
        k_orthogonal = k_vo
    elif args.case == "anisotropic_avg":
        k_parallel = 0.5 * (k_pp + k_vp)
        k_orthogonal = 0.5 * (k_po + k_vo)
    K = fem.Constant(msh, k_parallel * np.outer(d,d) + k_orthogonal * (np.eye(dim, dtype=default_real_type) - np.outer(d,d)))
root_print(f"Permeability: {K.value if isinstance(K, fem.Constant) else K}")

# Function spaces for the velocity and for the pressure
V = fem.functionspace(msh, ("Raviart-Thomas", args.order + 1))
Q = fem.functionspace(msh, ("Discontinuous Lagrange", args.order))
VQ = ufl.MixedFunctionSpace(V, Q)
root_print(f"Number of DOFs (V, Q): {V.dofmap.index_map.size_global}, {Q.dofmap.index_map.size_global}", flush=True)

# Define trial and test functions
u, p = ufl.TrialFunctions(VQ)
v, q = ufl.TestFunctions(VQ)

# Define pressure boundary conditions
p_in = fem.Function(Q)
p_in.interpolate(lambda x: np.ones_like(x[0])) # left BC: p=1, right BC: p=0
INLET = 1
inlet_facets = np.sort(mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)))
mt = mesh.meshtags(msh, msh.topology.dim - 1, inlet_facets, INLET)
outlet_facets = np.sort(mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.isclose(x[0], 1.0)))
# Define essential BCs on other domain boundaries (no flow)
def noflow_boundary(x):
    if dim==2:
        return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    else:
        return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) | np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)

wall_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, noflow_boundary)
test = wall_facets.size
wall_facets = np.setdiff1d(wall_facets, np.hstack((inlet_facets, outlet_facets)))  # remove inlet facets from wall facets
if len(wall_facets) < test:
    print(f"Rank {MPI.COMM_WORLD.rank}: Error in boundary detection: {test - len(wall_facets)} inlet facets detected as wall facets")
wall_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, wall_facets)
wall_bc = fem.dirichletbc(fem.Function(V), wall_dofs)  # zero velocity

# Define bilinear form
dx = ufl.Measure("dx", msh)
a_00 = ufl.inner(mu * ufl.inv(K) * u, v) * dx
a_01 = ufl.inner(p, ufl.div(v)) * dx
a_10 = ufl.inner(ufl.div(u), q) * dx
a_ufl = [[a_00, a_01], [a_10, None]]
a = fem.form(a_ufl)

# Define linear form
n = ufl.FacetNormal(msh)
ds = ufl.Measure("ds", msh, subdomain_data=mt)
L_0 = p_in * ufl.inner(v, n) * ds(subdomain_id=INLET)
L_1 = ufl.inner(fem.Constant(msh, default_real_type(0.0)), q) * dx
L_ufl = [L_0, L_1]
L = fem.form(L_ufl)

if DX_VERSION == "0.9.0":
    # Assemble Darcy problem
    A = assemble_matrix_block(a, bcs=[wall_bc]) # no essential BCs for velocity
    A.assemble()
    b = assemble_vector_block(L, a, bcs=[wall_bc]) # no essential BCs for pressure

    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)  # type: ignore
    ksp.setOperators(A)
    ksp.setTolerances(rtol=rel_tol, max_it=10)
    ksp.setType("richardson")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()  # type: ignore
    opts["ksp_monitor" ] = ""
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
    opts["ksp_error_if_not_converged"] = 1
    ksp.setFromOptions()

    x = A.createVecRight()
    ksp.solve(b, x)

    # Split the solution
    u_h, p_h = fem.Function(V), fem.Function(Q)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u_h.x.array[:offset] = -x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()

else:
    options = {
            "ksp_rtol": rel_tol,
            "ksp_monitor": "",
            "mat_mumps_icntl_14": 80, # Increase MUMPS working memory
            "mat_mumps_icntl_24": 1,  # Support solving a singular matrix (pressure nullspace)
            "mat_mumps_icntl_25": 0,  # Support solving a singular matrix (pressure nullspace)
    }

    if args.monolithic:
        options.update({
            "ksp_type": "richardson",
            "ksp_max_it": 10,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        })
        problem = LinearProblem(a_ufl, L_ufl, bcs=[wall_bc], petsc_options_prefix="darcy_", petsc_options=options)
    else:
        options.update({
            "ksp_type": "fgmres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "gkb",
            "pc_fieldsplit_gkb_monitor": "",
            "pc_fieldsplit_gkb_nu": mu**-2,
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "cholesky",
            "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        })
        problem = LinearProblem(a_ufl, L_ufl, kind="nest", bcs=[wall_bc], petsc_options_prefix="darcy_", petsc_options=options)

    u_h, p_h = problem.solve()
    # correct sign
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u_h.x.array[:offset] = -u_h.x.array[:offset]
    u_h.x.scatter_forward()
    assert problem.solver.getConvergedReason() > 0

# Function space for visualizing the velocity field
W = fem.functionspace(msh, ("Discontinuous Lagrange", args.order + 1, (dim,)))
u_vis = fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)
p_h.name = "p"

# Write solution to file
try:
    dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "darcy", f"cube_eps{args.epsilon}")
    os.makedirs(dir_name, exist_ok=True)
    u_file = io.VTXWriter(msh.comm, os.path.join(dir_name, f"{args.case}_velocity.bp"), [u_vis._cpp_object] if DX_VERSION == "0.9.0" else u_vis)
    p_file = io.VTXWriter(msh.comm, os.path.join(dir_name, f"{args.case}_pressure.bp"), [p_h._cpp_object] if DX_VERSION == "0.9.0" else p_h)
    u_file.write(0.0)
    u_file.close()
    p_file.write(0.0)
    p_file.close()
except AttributeError:
    print("File output requires ADIOS2.")

# Check divergence and flow
div_norm = MPI.COMM_WORLD.reduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.div(u_h), ufl.div(u_h)) * dx)))
if div_norm is not None:
    print(f"L2 norm of divergence(u): {np.sqrt(div_norm):.3e}")
total_inflow = MPI.COMM_WORLD.reduce(np.array([fem.assemble_scalar(fem.form(u_h[i] * ds(subdomain_id=INLET))) for i in range(dim)]))
average_flow = MPI.COMM_WORLD.reduce(np.array([fem.assemble_scalar(fem.form(u_h[i] * dx)) for i in range(dim)]))
volume = MPI.COMM_WORLD.reduce(fem.assemble_scalar(fem.form(1 * dx)))
if volume is not None:
    print(f"Total in/outflow: {total_inflow}\nAverage flow:     {[average_flow[i] / volume for i in range(dim)]}\nVolume: {volume}")