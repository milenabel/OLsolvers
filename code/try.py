import os
import json
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
from ufl import dx, grad, inner
import pandas as pd


class GeneralSolver:
    def __init__(self, domain_size=(12.0, 12.0), mesh_sizes=[(20, 20), (24, 24), (48, 48), (96, 96)],
                 results_dir="../results/general", rhs_base_dir="../results/manufactured"):
        self.domain_size = domain_size
        self.mesh_sizes = mesh_sizes
        self.results_dir = results_dir
        self.rhs_base_dir = rhs_base_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def solve_all(self):
        for mesh_size in self.mesh_sizes:
            Nx, Ny = mesh_size
            mesh_str = f"{Nx}x{Ny}"
            rhs_path = os.path.join(self.rhs_base_dir, mesh_str, f"rhs_coeffs_{mesh_str}.json")

            if not os.path.exists(rhs_path):
                print(f"[Warning] RHS file not found for mesh {mesh_str}. Skipping...")
                continue

            print(f"\n[Running] Solving for mesh {mesh_str}")
            self.solve_one_mesh(mesh_size, rhs_path)

    def solve_one_mesh(self, mesh_size, rhs_path):
        with open(rhs_path, "r") as f:
            rhs_coeffs = json.load(f)
        num_rhs = len(rhs_coeffs)

        msh = mesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), self.domain_size], mesh_size, mesh.CellType.quadrilateral)
        V = fem.functionspace(msh, ("Lagrange", 1))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)

        # Homogeneous Dirichlet BC
        zero = fem.Function(V)
        boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
        bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)
        bc = fem.dirichletbc(zero, bc_dofs)

        # Assemble matrix
        a_form = fem.form(inner(grad(u), grad(v)) * dx)
        A = fem.petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()

        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()

        f_array = []
        u_array = []
        error_metrics = []

        for i, a_k in enumerate(rhs_coeffs):
            u_expr = sum(a_k[k] * ufl.cos((k + 1) * np.pi * x[0] / self.domain_size[0]) *
                         ufl.sin((k + 1) * np.pi * x[1] / self.domain_size[1]) for k in range(len(a_k)))
            f_expr = -ufl.div(ufl.grad(u_expr))

            f_func = fem.Function(V)
            f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points()))

            # Assemble RHS
            L_form = fem.form(inner(f_func, v) * dx)
            b = fem.petsc.assemble_vector(L_form)
            apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve
            u_sol = fem.Function(V)
            x_vec = A.createVecRight()
            solver.solve(b, x_vec)
            u_sol.x.array[:] = x_vec.array

            f_array.append(f_func.x.array.copy())
            u_array.append(u_sol.x.array.copy())

            u_exact = fem.Function(V)
            u_exact.interpolate(fem.Expression(u_expr, V.element.interpolation_points()))

            l2_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(inner(u_sol, u_sol) * dx)), op=MPI.SUM))
            l2_error = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(inner(u_sol - u_exact, u_sol - u_exact) * dx)), op=MPI.SUM))
            h1_error = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(inner(grad(u_sol - u_exact), grad(u_sol - u_exact)) * dx)), op=MPI.SUM))
            relative_error = l2_error / l2_norm if l2_norm > 0 else 0.0

            error_metrics.append([i, l2_error, h1_error, relative_error])

        # Save dataset JSON
        Nx, Ny = mesh_size
        dataset = {
            "mesh_size": mesh_size,
            "domain_size": self.domain_size,
            "u_values": [u.tolist() for u in u_array],
            "f_values": [f.tolist() for f in f_array]
        }

        dataset_file = os.path.join(self.results_dir, f"poisson_dataset_{Nx}x{Ny}.json")
        error_file = os.path.join(self.results_dir, f"poisson_error_norms_{Nx}x{Ny}.csv")

        with open(dataset_file, "w") as f:
            json.dump(dataset, f)
        print(f"[Saved] FEM dataset -> {dataset_file}")

        error_df = pd.DataFrame(error_metrics, columns=["Sample", "L2 Error", "H1 Error", "Relative L2 Error"])
        error_df.to_csv(error_file, index=False)
        print(f"[Saved] Error metrics -> {error_file}")


if __name__ == "__main__":
    solver = GeneralSolver()
    solver.solve_all()
