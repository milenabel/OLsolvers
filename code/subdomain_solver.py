import os
import json
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner
from scipy.interpolate import LinearNDInterpolator
from tqdm import trange

def solve_subdomains_batched(global_mesh_size, domain_size, u_values_list, subdomain_bounds, sub_mesh_size, results_dir="../results/subdomains"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # === Reconstruct global mesh once ===
    msh_global = mesh.create_rectangle(comm, [(0.0, 0.0), domain_size], global_mesh_size, mesh.CellType.quadrilateral)
    global_coords = msh_global.geometry.x[:, :2]

    coords_sub = None
    u_values_all = []
    L2_norms = []
    H1_norms = []
    rel_L2_norms = []

    for sample_id in trange(len(u_values_list), desc=f"{global_mesh_size} → {sub_mesh_size}"):
        u_global = np.array(u_values_list[sample_id], dtype=np.float64)
        u_interp = LinearNDInterpolator(global_coords, u_global)

        # Create subdomain mesh
        msh_sub = mesh.create_rectangle(
            comm,
            points=((subdomain_bounds[0], subdomain_bounds[2]), (subdomain_bounds[1], subdomain_bounds[3])),
            n=sub_mesh_size,
            cell_type=mesh.CellType.quadrilateral,
        )
        V_sub = fem.functionspace(msh_sub, ("Lagrange", 1))
        u = ufl.TrialFunction(V_sub)
        v = ufl.TestFunction(V_sub)

        # Interpolate global solution on boundary
        facets = mesh.locate_entities_boundary(msh_sub, msh_sub.topology.dim - 1, lambda x: np.full(x.shape[1], True))
        dofs = fem.locate_dofs_topological(V_sub, msh_sub.topology.dim - 1, facets)

        bc_fn = fem.Function(V_sub)
        dof_coords = V_sub.tabulate_dof_coordinates()[dofs][:, :2]
        bc_vals = u_interp(dof_coords[:, 0], dof_coords[:, 1])
        bc_fn.x.array[dofs] = np.nan_to_num(bc_vals, nan=0.0)

        bc = fem.dirichletbc(bc_fn, dofs)

        # Solve Poisson
        a = inner(grad(u), grad(v)) * dx
        L = inner(fem.Constant(msh_sub, 0.0), v) * dx
        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        # Store error metrics
        L2 = np.sqrt(msh_sub.comm.allreduce(fem.assemble_scalar(fem.form(inner(uh - bc_fn, uh - bc_fn) * dx)), op=MPI.SUM))
        H1 = np.sqrt(msh_sub.comm.allreduce(fem.assemble_scalar(fem.form(inner(grad(uh - bc_fn), grad(uh - bc_fn)) * dx)), op=MPI.SUM))
        proj = np.sqrt(msh_sub.comm.allreduce(fem.assemble_scalar(fem.form(inner(bc_fn, bc_fn) * dx)), op=MPI.SUM))
        rel = L2 / proj if proj > 0 else 0.0

        u_values_all.append(uh.x.array.tolist())
        L2_norms.append(float(L2))
        H1_norms.append(float(H1))
        rel_L2_norms.append(float(rel))

        if coords_sub is None:
            coords_sub = msh_sub.geometry.x[:, :2].tolist()

    if rank == 0:
        out = {
            "global_mesh": global_mesh_size,
            "domain_size": domain_size,
            "subdomain": subdomain_bounds,
            "sub_mesh": sub_mesh_size,
            "coords": coords_sub,
            "u_values": u_values_all,
            "L2_norms": L2_norms,
            "H1_norms": H1_norms,
            "Relative_L2_norms": rel_L2_norms
        }
        Nx, Ny = global_mesh_size
        sx, sy = sub_mesh_size
        save_path = f"{results_dir}/fem_subdomain_g{Nx}x{Ny}_s{sx}x{sy}.json"
        os.makedirs(results_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(out, f)
        print(f"[Saved] {save_path}")

if __name__ == "__main__":
    # global_mesh_sizes = [(20, 20), (24, 24), (48, 48), (96, 96)]
    global_mesh_sizes = [(20, 20)]
    subdomain_refinements = [
        (3, 3), (4, 4), (6, 6), (8, 8), (12, 12), (16, 16),
        (20, 20), (24, 24), (32, 32), (48, 48), (64, 64),
        (96, 96), (128, 128), (192, 192)
    ]
    subdomains = [(0, 4, 0, 4)]

    for gsize in global_mesh_sizes:
        Nx, Ny = gsize
        fpath = f"../results/general/poisson_dataset_{Nx}x{Ny}.json"
        if not os.path.exists(fpath):
            print(f"[Skip] No dataset for {Nx}x{Ny}")
            continue

        with open(fpath, "r") as f:
            data = json.load(f)

        domain_size = tuple(data["domain_size"])
        u_values_list = data["u_values"]

        for subdomain in subdomains:
            for sub_mesh in subdomain_refinements:
                solve_subdomains_batched(
                    global_mesh_size=gsize,
                    domain_size=domain_size,
                    u_values_list=u_values_list,
                    subdomain_bounds=subdomain,
                    sub_mesh_size=sub_mesh,
                    results_dir="../results/subdomains"
                )
