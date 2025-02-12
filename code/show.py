from mpi4py import MPI
import pyvista
import numpy as np
from dolfinx import io, fem, plot, mesh

# Load the stored mesh WITHOUT specifying a name
with io.XDMFFile(MPI.COMM_WORLD, "poisson_generalized.xdmf", "r") as file:
    msh = file.read_mesh()
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)  # Ensure connectivity

# Create function space
V = fem.functionspace(msh, ("Lagrange", 1))

# Read function data manually
uh = fem.Function(V)
with io.XDMFFile(MPI.COMM_WORLD, "poisson_generalized.xdmf", "r") as file:
    file.read_function(uh)

# Convert function to PyVista-compatible format
cells, cell_types, points = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(np.hstack(cells), cell_types, points)

# Attach the solution values to the mesh
grid.point_data["Solution"] = uh.x.array.real  # Ensure it's real-valued
grid.set_active_scalars("Solution")

# Create a PyVista plot
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)

# Warp the solution for better visualization
warped = grid.warp_by_scalar()
plotter.add_mesh(warped, cmap="coolwarm")

# Show or save visualization
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)
    plotter.screenshot("uh_poisson.png")
else:
    plotter.show()
