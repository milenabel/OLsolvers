#!/bin/bash

# Exit on error
set -e

echo "Activating fenicsx-env..."
eval "$(conda shell.bash hook)"
conda activate fenicsx-env

echo "Running data_gen_meshes.py in fenicsx-env..."
python3 data_gen_meshes.py

echo "Running general_solver_meshes.py in fenicsx-env..."
python3 general_solver_meshes.py

# echo "Switching to jax-env..."
# conda activate jax-env

# echo "Running deeponet.py in jax-env..."
# python3 deeponet.py

echo "All done!"
