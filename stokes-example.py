# Divergence-free 3D Stokes solver using RT-DG or BDM-DG elements
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
parser.add_argument("-a", "--alpha", type=float, default=10.0, help="Interior penalty parameter")
parser.add_argument("--MG", action=argparse.BooleanOptionalAction, default=False, help="Use multigrid solver inside the block solver")
parser.add_argument("-m", "--monolithic", action=argparse.BooleanOptionalAction, default=False, help="Use monolithic solver instead of block solver")
parser.add_argument("-n", "--nx", type=int, default=16, help="Number of vertices per dim: dx = 1/nx")
parser.add_argument("-o", "--order", type=int, default=0, help="FEM order")
parser.add_argument("--BDM", action=argparse.BooleanOptionalAction, default=False, help="Use BDM elements instead of RT")
args = parser.parse_args()
root_print(f"Options used: {args}", flush=True)

# Mesh and parameters
mu = 1e-3  # viscosity
rel_tol = 1e-12 # Solver relative tolerance

msh = mesh.create_unit_cube(MPI.COMM_WORLD, args.nx, args.nx, args.nx)
dim = msh.geometry.dim

# Function spaces for the velocity and for the pressure
if args.BDM:
    V = fem.functionspace(msh, ("BDM", args.order + 1))
else:
    V = fem.functionspace(msh, ("Raviart-Thomas", args.order + 1))
Q = fem.functionspace(msh, ("Discontinuous Lagrange", args.order))

# Function space for visualizing the velocity field
W = fem.functionspace(msh, ("Discontinuous Lagrange", args.order + 1, (msh.geometry.dim,)))

# Define trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
root_print(f"Number of DOFs (V, Q): {V.dofmap.index_map.size_global}, {Q.dofmap.index_map.size_global}", flush=True)

alpha = fem.Constant(msh, default_real_type(args.alpha * (args.order + 1)**2))

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

def jump(phi, n):
    return ufl.outer(phi("+"), n("+")) + ufl.outer(phi("-"), n("-"))

# Define pressure boundary conditions
p_in = fem.Function(Q)
p_in.interpolate(lambda x: np.ones_like(x[0])) # left BC: p=1, right BC: p=0

INLET = 1
inlet_facets = np.sort(mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)))
mt_in = mesh.meshtags(msh, msh.topology.dim - 1, inlet_facets, INLET)
OUTLET = 2
outlet_facets = np.sort(mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.isclose(x[0], 1.0)))
mt_out = mesh.meshtags(msh, msh.topology.dim - 1, outlet_facets, OUTLET)

# Define essential BCs on other domain boundaries (no flow)
def noflow_boundary(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) | np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)
WALL = 3
wall_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, noflow_boundary)
test = wall_facets.size
wall_facets = np.setdiff1d(wall_facets, np.hstack((inlet_facets, outlet_facets)))  # remove inlet facets from wall facets
if len(wall_facets) < test:
    print(f"Rank {MPI.COMM_WORLD.rank}: Error in boundary detection: {test - len(wall_facets)} inlet facets detected as wall facets")
mt_wall = mesh.meshtags(msh, msh.topology.dim - 1, wall_facets, WALL)

wall_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, wall_facets)
wall_bc = fem.dirichletbc(fem.Function(V), wall_dofs)  # zero velocity

dx = ufl.Measure("dx", msh)
dS = ufl.Measure("dS", msh)
ds_w = ufl.Measure("ds", msh, subdomain_data=mt_wall, subdomain_id=WALL)

a_00 = mu * (ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
             # Interior penalty terms
             - ufl.inner(ufl.avg(ufl.grad(u)), jump(v, n)) * dS
             - ufl.inner(jump(u, n), ufl.avg(ufl.grad(v))) * dS
             + (alpha / ufl.avg(h)) * ufl.inner(jump(u, n), jump(v, n)) * dS
             # Nitsche terms for Dirichlet BCs
             - ufl.inner(ufl.grad(u), ufl.outer(v, n)) * ds_w
             - ufl.inner(ufl.outer(u, n), ufl.grad(v)) * ds_w
             + (alpha / h) * ufl.inner(ufl.outer(u, n), ufl.outer(v, n)) * ds_w)
a_01 = - ufl.inner(p, ufl.div(v)) * dx
a_10 = - ufl.inner(ufl.div(u), q) * dx

a_ufl = [[a_00, a_01], [a_10, None]]
a = fem.form(a_ufl)

ds_in = ufl.Measure("ds", msh, subdomain_data=mt_in, subdomain_id=INLET)
L_0 = - p_in * ufl.inner(v, n) * ds_in
L_1 = ufl.inner(fem.Constant(msh, default_real_type(0.0)), q) * dx
L_ufl = [L_0, L_1]
L = fem.form(L_ufl)

# Boundary conditions
bcs = [wall_bc]

if DX_VERSION == "0.9.0":
    # Assemble Stokes problem
    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    b = assemble_vector_block(L, a, bcs=bcs)

    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)  # type: ignore
    ksp.setOperators(A)
    opts = PETSc.Options()  # type: ignore
    opts["ksp_monitor" ] = ""
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
    opts["ksp_error_if_not_converged"] = 1
    ksp.setFromOptions()
    ksp.setType("richardson")
    ksp.setTolerances(rtol=rel_tol, max_it=10)
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    # Solve Stokes for initial condition
    x = A.createVecRight()
    ksp.solve(b, x)

    # Split the solution
    u_h, p_h = fem.Function(V), fem.Function(Q)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u_h.x.array[:offset] = x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()

else: # DX_VERSION == 0.10.0
    options = {
        "ksp_rtol": rel_tol,
        "ksp_monitor": "",
        "mat_mumps_icntl_14": 80, # Increase MUMPS working memory
    }
    if args.monolithic:
        options.update({
            "ksp_type": "richardson",
            "ksp_max_it": 10,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        })
        problem = LinearProblem(a_ufl, L_ufl, bcs=bcs, petsc_options_prefix="stokes_", petsc_options=options)
    else:
        options.update({
            "ksp_type": "fgmres",
            #"ksp_view": "",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "gkb",
            "pc_fieldsplit_gkb_monitor": "",
            "pc_fieldsplit_gkb_nu": mu**-2,
        })
        if args.MG:
            options.update({
                "fieldsplit_0_ksp_type": "cg",
                "fieldsplit_0_pc_type": "gamg",
            })
        else:
            options.update({
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "cholesky",
                "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
            })
        problem = LinearProblem(a_ufl, L_ufl, kind="nest", bcs=bcs, petsc_options_prefix="stokes_", petsc_options=options)

        A00 = problem.A.getNestSubMatrix(0, 0)
        A00.setOption(PETSc.Mat.Option.SPD, True)

    u_h, p_h = problem.solve()
    assert problem.solver.getConvergedReason() > 0

p_h.name = "p"
u_vis = fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)

# Write initial condition to file
try:
    dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stokes", "example")
    os.makedirs(dir_name, exist_ok=True)
    u_file = io.VTXWriter(msh.comm, os.path.join(dir_name, "velocity.bp"), [u_vis._cpp_object] if DX_VERSION == "0.9.0" else u_vis)
    p_file = io.VTXWriter(msh.comm, os.path.join(dir_name, "pressure.bp"), [p_h._cpp_object] if DX_VERSION == "0.9.0" else p_h)
    u_file.write(0.0)
    u_file.close()
    p_file.write(0.0)
    p_file.close()
except AttributeError:
    print("File output requires ADIOS2.")

# 9. Check divergence and pressure
div_norm = MPI.COMM_WORLD.reduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.div(u_h), ufl.div(u_h)) * dx)))
if div_norm is not None:
    print(f"L2 norm of divergence(u): {np.sqrt(div_norm):.3e}")
p_avg_in = MPI.COMM_WORLD.reduce(fem.assemble_scalar(fem.form(p_h * ds_in)))
if p_avg_in is not None:
    print(f"Average inlet pressure error: {p_avg_in-1:.3e}")
ds_out = ufl.Measure("ds", msh, subdomain_data=mt_out, subdomain_id=OUTLET)
p_avg_out = MPI.COMM_WORLD.reduce(fem.assemble_scalar(fem.form(p_h * ds_out)))
if p_avg_out is not None:
    print(f"Average outlet pressure error: {p_avg_out:.3e}")