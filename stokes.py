# Divergence-free 3D Stokes solver using RT-DG / BDM-DG elements
# (requires dolfinx 0.9.0 or 0.10.0)
#
# Author: S.B. Lunowa

import argparse
import numpy as np
import os

from dolfinx import default_real_type, fem, io, la, mesh
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
parser.add_argument("-c", "--case", choices=["cube_cylinder", "cylinder_spiral", "artificial_vessel"], default="cube_cylinder")
parser.add_argument("-o", "--order", type=int, default=1, help="FEM order")
parser.add_argument('--BDM', action=argparse.BooleanOptionalAction, default=False, help="Whether to use BDM instead of RT finite element space")
parser.add_argument('--TH', action=argparse.BooleanOptionalAction, default=False, help="Whether to use Taylor-Hood instead of RT finite element space")
parser.add_argument('--cholesky', action=argparse.BooleanOptionalAction, default=False, help="Whether to use a Cholesky factorization in the preconditioner.")
args = parser.parse_args()
root_print(f"Options used: {args}", flush=True)

if args.TH and args. BDM:
    ValueError("Cannot use BDM and TH elements at the same time...")
if args.TH:
    root_print("Warning: The Taylor-Hood element is not divergence free!")

current_dir = os.path.dirname(os.path.abspath(__file__))
if args.case == "cube_cylinder":
    if DX_VERSION == "0.9.0":
        msh, cell_tags, facet_tags = io.gmshio.read_from_msh(os.path.join(current_dir, "geometry", f"{args.case}.msh"),
                                                             MPI.COMM_WORLD, partitioner=mesh.create_cell_partitioner(mesh.GhostMode.shared_facet))
    else:
        msh, cell_tags, facet_tags, *_ = io.gmsh.read_from_msh(os.path.join(current_dir, "geometry", f"{args.case}.msh"),
                                                               MPI.COMM_WORLD, partitioner=mesh.create_cell_partitioner(mesh.GhostMode.shared_facet))
else:
    if DX_VERSION == "0.9.0":
        msh, cell_tags, facet_tags = io.gmshio.read_from_msh(os.path.join(current_dir, "geometry", args.case, "domain_cut_3d.msh"),
                                                             MPI.COMM_WORLD, partitioner=mesh.create_cell_partitioner(mesh.GhostMode.shared_facet))
    else:
        msh, cell_tags, facet_tags, *_ = io.gmsh.read_from_msh(os.path.join(current_dir, "geometry", args.case, "domain_cut_3d.msh"),
                                                               MPI.COMM_WORLD, partitioner=mesh.create_cell_partitioner(mesh.GhostMode.shared_facet))

# print(f"Facet tags: {np.unique(facet_tags.values)}")
INLET  = 1
OUTLET = 2
WALL   = 3

wall_facets   = facet_tags.find(WALL)
outlet_facets = facet_tags.find(OUTLET)
inlet_facets  = facet_tags.find(INLET)

rel_tol = 1e-12      # Solver relative tolerance
if args.TH:
    rel_tol = 1e-6
mu = fem.Constant(msh, default_real_type(1e-3)) # viscosity

# Function spaces for the velocity and for the pressure
if args.TH:
    V = fem.functionspace(msh, ("Lagrange", args.order + 1, (msh.geometry.dim,)))
    Q = fem.functionspace(msh, ("Lagrange", args.order))
else:
    V = fem.functionspace(msh, ("BDM" if args.BDM else "RT", args.order + 1))
    Q = fem.functionspace(msh, ("DG", args.order))

root_print(f"Number of DOFs (V, Q): {V.dofmap.index_map.size_global}, {Q.dofmap.index_map.size_global}", flush=True)

# Define trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

alpha = fem.Constant(msh, default_real_type(25 * (args.order + 1)**2))

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

def jump(phi, n):
    return ufl.outer(phi("+"), n("+")) + ufl.outer(phi("-"), n("-"))

dx = ufl.Measure("dx", msh)
dS = ufl.Measure("dS", msh)
ds = ufl.Measure("ds", msh, subdomain_data=facet_tags)
ds_w = ds(subdomain_data=facet_tags, subdomain_id=WALL)

if args.TH:
    a_00 = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx #+ alpha * ufl.inner(ufl.div(u), ufl.div(v)) * dx
else:
    a_00 = mu * (ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
                # Interior penalty terms
                - ufl.inner(ufl.avg(ufl.grad(u)), jump(v, n)) * dS
                - ufl.inner(jump(u, n), ufl.avg(ufl.grad(v))) * dS
                + (alpha / ufl.avg(h)) * ufl.inner(jump(u, n), jump(v, n)) * dS
                # Nitsche terms for inlet and wall (no slip)
                - ufl.inner(ufl.grad(u), ufl.outer(v, n)) * ds_w
                - ufl.inner(ufl.outer(u, n), ufl.grad(v)) * ds_w
                + (alpha / h) * ufl.inner(ufl.outer(u, n), ufl.outer(v, n)) * ds_w)
a_01 = - ufl.inner(p, ufl.div(v)) * dx
a_10 = - ufl.inner(ufl.div(u), q) * dx

a_ufl = [[a_00, a_01], [a_10, None]]
a = fem.form(a_ufl)

# Define form for diagonal preconditioning
a_p11 = fem.form(ufl.inner(p, q) / mu * dx)
a_p = [[a[0][0], None], [None, a_p11]]

# Define boundary conditions and linear form
wall_dofs  = fem.locate_dofs_topological(V, msh.topology.dim - 1, wall_facets)
bc_wall  = fem.dirichletbc(fem.Function(V), wall_dofs)  # zero velocity
bcs = [bc_wall]
# inflow pressure is weakly imposed, homogeneous Neumann at outflow, i.e., no linear term necessary,
# and no contribution of wall as velocity is zero
p_in = fem.Constant(msh, default_real_type(1.0))
ds_in = ds(subdomain_data=facet_tags, subdomain_id=INLET)
L_0 = -p_in * ufl.inner(v, n) * ds_in
L_1 = ufl.inner(fem.Constant(msh, default_real_type(0.0)), q) * dx
L_ufl = [L_0, L_1]
L = fem.form(L_ufl)

# Create FE Functions to hold the solution
u_h, p_h = fem.Function(V), fem.Function(Q)

# Create and configure solver
if DX_VERSION == "0.9.0":
    # Assemble nested matrix operators
    A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
    A.assemble()

    # Create a nested matrix P to use as the preconditioner. The
    # top-left block of P is shared with the top-left block of A. The
    # bottom-right diagonal entry is assembled from the form a_p11:
    P11 = fem.petsc.assemble_matrix(a_p11, [])
    P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
    P.assemble()

    A00 = A.getNestSubMatrix(0, 0)
    A00.setOption(PETSc.Mat.Option.SPD, True)

    P00, P11 = P.getNestSubMatrix(0, 0), P.getNestSubMatrix(1, 1)
    P00.setOption(PETSc.Mat.Option.SPD, True)
    P11.setOption(PETSc.Mat.Option.SPD, True)

    # Assemble right-hand side vector
    b = fem.petsc.assemble_vector_nest(L)

    # Modify ('lift') the RHS for Dirichlet boundary conditions
    fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

    # Sum contributions for vector entries that are shared across parallel processes
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS vector
    bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L), bcs)
    fem.petsc.set_bc_nest(b, bcs0)

    # If the pressure field is determined only up to a constant, we supply a vector that
    # spans the nullspace to the solver, and any component of the solution in this direction
    # will be eliminated during the solution process.
    null_vec = fem.petsc.create_vector_nest(L)
    # Set velocity part to zero and the pressure part to a non-zero constant
    null_vecs = null_vec.getNestSubVecs()
    null_vecs[0].set(0.0), null_vecs[1].set(1.0)
    # Normalize the vector that spans the nullspace, create a nullspace object, and attach it to the matrix
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    if nsp.test(A):
        A.setNullSpace(nsp)

    # Create a MINRES Krylov solver and a block-diagonal preconditioner using PETSc's fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    options = PETSc.Options()
    options["ksp_monitor" ] = ""
    options["ksp_error_if_not_converged"] = ""
    options["mat_mumps_icntl_14"] = 800 # Increase MUMPS working memory
    options["mat_mumps_icntl_24"] = 1   # Support solving a singular matrix (pressure nullspace)
    options["mat_mumps_icntl_25"] = 0   # Support solving a singular matrix (pressure nullspace)
    if args.split == "SCHUR":
        options["pc_fieldsplit_off_diag_use_amat"] = ""
    ksp.setFromOptions()
    ksp.setType("minres")
    ksp.setTolerances(rtol=rel_tol)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    # Define the matrix blocks in the preconditioner with the velocity and pressure matrix index sets
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))
    ksp.getPC().setUp()

    # Set the preconditioners for each block.
    # By default, GAMG will infer the correct near-nullspace from the matrix block size.
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_p.setType("preonly")
    if args.cholesky:
        ksp_u.getPC().setType("cholesky")
        ksp_u.getPC().setFactorSolverType("mumps")
        ksp_u.getPC().setFactorShift(shift_type=PETSc.Mat.FactorShiftType.POSITIVE_DEFINITE, amount=1e-9)
        ksp_p.getPC().setType("gamg")
    else:
        ksp_u.getPC().setType("gamg")
        ksp_p.getPC().setType("jacobi")

    # The vectors for `u` and `p` are combined to form a nested vector and the system is solved.
    x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(u_h.x), la.create_petsc_vector_wrap(p_h.x)])
    ksp.solve(b, x)
    u_h.x.scatter_forward()
    p_h.x.scatter_forward()

else: # DX_VERSION == "0.10.0"
    # Solve the Stokes problem using a nested PETSc iterative solver
    options = {
        "ksp_type": "fgmres",
        "ksp_rtol": rel_tol,
        "ksp_monitor": "",
        #"ksp_view": "",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "gkb",
        "pc_fieldsplit_gkb_monitor": "",
        "pc_fieldsplit_gkb_nu": 1e6,
    }

    if args.cholesky:
        options.update({
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "cholesky",
            "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 800, # Increase MUMPS working memory
        })
    else:
        options.update({
            "fieldsplit_0_ksp_type": "cg",
            "fieldsplit_0_pc_type": "gamg",
        })

    problem = LinearProblem(a_ufl, L_ufl, kind="nest", bcs=bcs, petsc_options_prefix="stokes_", petsc_options=options)

    A00 = problem.A.getNestSubMatrix(0, 0)
    A00.setOption(PETSc.Mat.Option.SPD, True)

    u_h, p_h = problem.solve()
    assert problem.solver.getConvergedReason() > 0

if not args.TH:
    # Function space for visualizing the velocity field
    W = fem.functionspace(msh, ("DG", args.order + 1, (msh.geometry.dim,)))
    u_vis = fem.Function(W)
    u_vis.interpolate(u_h)
else:
    u_vis = u_h
u_vis.name = "u"
p_h.name = "p"

# Write solution to file
try:
    dir_name = os.path.join(current_dir, "stokes", args.case)
    if args.TH:
        dir_name = os.path.join(dir_name, "taylor-hood")
    os.makedirs(dir_name, exist_ok=True)
    u_file = io.VTXWriter(msh.comm, os.path.join(dir_name, "velocity.bp"), [u_vis._cpp_object] if DX_VERSION == "0.9.0" else u_vis)
    u_file.write(0.0)
    u_file.close()
    p_file = io.VTXWriter(msh.comm, os.path.join(dir_name, "pressure.bp"), [p_h._cpp_object] if DX_VERSION == "0.9.0" else p_h)
    p_file.write(0.0)
    p_file.close()
except AttributeError:
    print("File output requires ADIOS2.")

# Check divergence and flow
div_norm = MPI.COMM_WORLD.reduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.div(u_h), ufl.div(u_h)) * dx)))
if div_norm is not None:
    print(f"L2 norm of divergence(u): {np.sqrt(div_norm):.3e}")
total_inflow = MPI.COMM_WORLD.reduce(np.array([fem.assemble_scalar(fem.form(u_h[i] * ds(subdomain_id=INLET))) for i in range(msh.geometry.dim)]))
total_outflow = MPI.COMM_WORLD.reduce(np.array([fem.assemble_scalar(fem.form(u_h[i] * ds(subdomain_id=OUTLET))) for i in range(msh.geometry.dim)]))
average_flow = MPI.COMM_WORLD.reduce(np.array([fem.assemble_scalar(fem.form(u_h[i] * dx)) for i in range(msh.geometry.dim)]))
volume = MPI.COMM_WORLD.reduce(fem.assemble_scalar(fem.form(1 * dx)))
if volume is not None:
    print(f"Total inflow:  {total_inflow}\nTotal outflow: {total_outflow}\nAverage flow:  {[average_flow[i] / volume for i in range(msh.geometry.dim)]}\nVolume: {volume}")