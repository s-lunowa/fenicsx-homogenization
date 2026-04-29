# Mixed RT-DG elements for Darcy flow
# (requires dolfinx 0.9.0 or 0.10.0)
#
# Author: S.B. Lunowa

import argparse
import numpy as np
import os

from dolfinx import default_real_type, fem, io
from dolfinx import __version__ as DX_VERSION
if DX_VERSION == "0.9.0":
    from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
else:
    from dolfinx.fem.petsc import LinearProblem
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from permeability.interpolatePointCloud import get_vtk_bounding_box, interpolate_vtk_field

def root_print(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

# Parameters and mesh
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--case", choices=["isotropic", "anisotropic", "anisotropic_p", "anisotropic_v", "anisotropic_avg"], default="isotropic")
parser.add_argument("-b", "--boundary", choices=["avg", "fluid", "reflection", "wall"], default="wall", help="Boundary value treatment")
parser.add_argument("-f", "--filter", choices=["box", "gaussian"], default="box", help="Averaging filter.")
parser.add_argument("-m", "--monolithic", action=argparse.BooleanOptionalAction, default=False, help="Use monolithic solver instead of block solver")
parser.add_argument("-r", "--radius", default="0.25", help="Radius of the averaging window.")
parser.add_argument("-o", "--order", type=int, default=1, help="FEM order")
parser.add_argument('--recentered', action=argparse.BooleanOptionalAction, default=False, help="Whether to recenter the porosity / direction fields.")
args = parser.parse_args()
root_print(f"Options used: {args}", flush=True)

radius = 0.25
mu = 1e-3 # dynamic viscosity
rel_tol = 2e-14
current_dir = os.path.dirname(os.path.abspath(__file__))
dir_name = os.path.join(current_dir, "darcy", f"cylinder_{args.case}")
os.makedirs(dir_name, exist_ok=True)
file_middle = f"REV_{args.radius}_Filter_{args.filter}_BC_{args.boundary}_cloud" # without cloud for cell centered data
fileprefix = os.path.join(dir_name, file_middle)

geometry_file = os.path.join(current_dir, "geometry", "cylinder.msh")
if os.path.exists(geometry_file):
    print(f"Warning: using separate mesh \"{geometry_file}\".")
    if DX_VERSION == "0.9.0":
        msh, cell_tags, facet_tags = io.gmshio.read_from_msh(geometry_file, MPI.COMM_WORLD)
    else:
        msh, cell_tags, facet_tags, *_ = io.gmsh.read_from_msh(geometry_file, MPI.COMM_WORLD)
else:
    if DX_VERSION == "0.9.0":
        msh, cell_tags, facet_tags = io.gmshio.read_from_msh(os.path.join(current_dir, "geometry", "cylinder_spiral", "domain_3d.msh"), MPI.COMM_WORLD)
    else:
        msh, cell_tags, facet_tags, *_ = io.gmsh.read_from_msh(os.path.join(current_dir, "geometry", "cylinder_spiral", "domain_3d.msh"), MPI.COMM_WORLD)
dim = msh.geometry.dim
INLET  = 1
OUTLET = 2
WALL   = 3
wall_facets   = facet_tags.find(WALL)
outlet_facets = facet_tags.find(OUTLET)
inlet_facets  = facet_tags.find(INLET)

# porosity file for permeability
porosity_file = os.path.join(current_dir, "permeability", "data", "cylinder_spiral", file_middle + ".vtk")
root_print(porosity_file)
P = fem.functionspace(msh, ("Discontinuous Lagrange", args.order))
porosity = fem.Function(P)
porosity.name = "porosity"
if args.recentered:
    bounds, center = get_vtk_bounding_box(porosity_file)
    center[0] -= 1.5
    print(f"Shift box {bounds} by {-center}.")
    porosity.interpolate(lambda x : interpolate_vtk_field(porosity_file, x.T, "porosity", translation=-center).T)
else:
    porosity.interpolate(lambda x : interpolate_vtk_field(porosity_file, x.T, "porosity").T)

if args.case == "isotropic":
    K = fem.Function(P)
    K.name = "permeability"
    K_expr = (2.0 * radius)**2 * porosity**3 / (150 * (1-porosity)**2) # isotropic permeability
    # TODO: limit to rasonable values
    K_expr2 = ufl.conditional(ufl.le(K_expr, 1e-12), 1e-12, ufl.conditional(ufl.le(K_expr, 1), K_expr, 1))
    if DX_VERSION == "0.9.0":
        K_fem = fem.Expression(K_expr2, P.element.interpolation_points())
    else:
        K_fem = fem.Expression(K_expr2, P.element.interpolation_points)
    K.interpolate(K_fem)
else:
    K_p = fem.Function(P)
    K_p.name = "permeability_parallel"
    K_o = fem.Function(P)
    K_o.name = "permeability_orthogonal"

    k_pp = -radius**2 / (4 * (1-porosity)) * (ufl.ln(1-porosity) + 0.5 * (1 - (1-porosity)) * (3 - (1-porosity)))
    k_po = -radius**2 / (8 * (1-porosity)) * (ufl.ln(1-porosity) + (1 - (1-porosity)**2) / (1 + (1-porosity)**2))
    k_vp = -radius**2 / (4 * (1-porosity)) * (ufl.ln(1-porosity) + 2 * (1 - (1-porosity)) / (1 + (1-porosity)))
    k_vo = -radius**2 / (8 * (1-porosity)) * (ufl.ln(1-porosity) + 2 * (1 - (1-porosity)) / (1 + (1-porosity)))

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

    # TODO: limit to rasonable values
    k_p_expr = ufl.conditional(ufl.le(k_parallel, 1e-12), 1e-12, ufl.conditional(ufl.le(k_parallel, 1), k_parallel, 1))
    k_o_expr = ufl.conditional(ufl.le(k_orthogonal, 1e-12), 1e-12, ufl.conditional(ufl.le(k_orthogonal, 1), k_orthogonal, 1))
    if DX_VERSION == "0.9.0":
        K_p.interpolate(fem.Expression(k_p_expr, P.element.interpolation_points()))
        K_o.interpolate(fem.Expression(k_o_expr, P.element.interpolation_points()))
    else:
        K_p.interpolate(fem.Expression(k_p_expr, P.element.interpolation_points))
        K_o.interpolate(fem.Expression(k_o_expr, P.element.interpolation_points))

    # K = k_p * tt^T + k_o * (I - tt^T) where t is the (tangential) coil direction
    T_space = fem.functionspace(msh, ("Discontinuous Lagrange", args.order, (dim, )))
    T = fem.Function(T_space)
    if args.recentered:
        T.interpolate(lambda x : interpolate_vtk_field(porosity_file, x.T, "wireDirection", translation=-center).T)
    else:
        T.interpolate(lambda x : interpolate_vtk_field(porosity_file, x.T, "wireDirection").T)
    K_space = fem.functionspace(msh, ("Discontinuous Lagrange", args.order, (dim, dim)))
    K = fem.Function(K_space)
    # K = k_p * tt^T + k_o * (I - tt^T) where t is the (tangential) coil direction
    if DX_VERSION == "0.9.0":
        K.interpolate(fem.Expression(K_o * (ufl.Identity(dim) - ufl.outer(T, T)) + K_p * ufl.outer(T, T), K_space.element.interpolation_points()))
    else:
        K.interpolate(fem.Expression(K_o * (ufl.Identity(dim) - ufl.outer(T, T)) + K_p * ufl.outer(T, T), K_space.element.interpolation_points))

# Write porosity and permeability to file
if args.case == "isotropic":
    obj_out = [porosity._cpp_object, K._cpp_object] if DX_VERSION == "0.9.0" else [porosity, K]
else:
    obj_out = [porosity._cpp_object, K_p._cpp_object, K_o._cpp_object] if DX_VERSION == "0.9.0" else [porosity, K_p, K_o]
with io.VTXWriter(msh.comm, fileprefix + "_param.bp", obj_out) as writer:
    writer.write(0.0)
    writer.close()

# Function spaces for the velocity and for the pressure
V = fem.functionspace(msh, ("Raviart-Thomas", args.order + 1))
Q = fem.functionspace(msh, ("Discontinuous Lagrange", args.order))
VQ = ufl.MixedFunctionSpace(V, Q)
root_print(f"Number of DOFs (V, Q): {V.dofmap.index_map.size_global}, {Q.dofmap.index_map.size_global}", flush=True)

# Define trial and test functions
u, p = ufl.TrialFunctions(VQ)
v, q = ufl.TestFunctions(VQ)

# Define bilinear form
dx = ufl.Measure("dx", msh)
a_00 = ufl.inner(mu * ufl.inv(K) * u, v) * dx
a_01 = ufl.inner(p, ufl.div(v)) * dx
a_10 = ufl.inner(ufl.div(u), q) * dx
a_ufl = [[a_00, a_01], [a_10, None]]
a = fem.form(a_ufl)

# Define pressure boundary conditions
p_in = fem.Function(Q)
p_in.interpolate(lambda x: np.ones_like(x[0])) # left BC: p=1, right BC: p=0
# Define essential BCs on other domain boundaries (no flow)
wall_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, wall_facets)
wall_bc = fem.dirichletbc(fem.Function(V), wall_dofs)  # zero velocity
# Define linear form
n = ufl.FacetNormal(msh)
ds = ufl.Measure("ds", msh, subdomain_data=facet_tags)
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
    u_h = fem.Function(V)
    p_h = fem.Function(Q)
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
    u_file = io.VTXWriter(msh.comm, fileprefix + "_velocity.bp", [u_vis._cpp_object] if DX_VERSION == "0.9.0" else u_vis)
    u_file.write(0.0)
    u_file.close()
    p_file = io.VTXWriter(msh.comm, fileprefix + "_pressure.bp", [p_h._cpp_object] if DX_VERSION == "0.9.0" else p_h)
    p_file.write(0.0)
    p_file.close()
except AttributeError:
    print("File output requires ADIOS2.")

# Check divergence and flow
div_norm = MPI.COMM_WORLD.reduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.div(u_h), ufl.div(u_h)) * dx)))
if div_norm is not None:
    print(f"L2 norm of divergence(u): {np.sqrt(div_norm):.3e}")
total_inflow = MPI.COMM_WORLD.reduce(np.array([fem.assemble_scalar(fem.form(u_h[i] * ds(subdomain_id=INLET))) for i in range(dim)]))
total_outflow = MPI.COMM_WORLD.reduce(np.array([fem.assemble_scalar(fem.form(u_h[i] * ds(subdomain_id=OUTLET))) for i in range(dim)]))
average_flow = MPI.COMM_WORLD.reduce(np.array([fem.assemble_scalar(fem.form(u_h[i] * dx)) for i in range(dim)]))
volume = MPI.COMM_WORLD.reduce(fem.assemble_scalar(fem.form(1 * dx)))
if volume is not None:
    print(f"Total inflow:  {total_inflow}\nTotal outflow: {total_outflow}\nAverage flow:  {[average_flow[i] / volume for i in range(dim)]}\nVolume: {volume}")
