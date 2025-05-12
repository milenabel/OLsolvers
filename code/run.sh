# #!/bin/bash

# # Exit on error
# set -e

# echo "Activating fenicsx-env..."
# # eval "$(conda shell.bash hook)"
# # conda activate fenicsx-env

# echo "Running data_gen_meshes.py in fenicsx-env..."
# python3 data_gen_meshes.py

# echo "Running general_solver_meshes.py in fenicsx-env..."
# python3 general_solver_meshes.py

# # echo "Switching to jax-env..."
# # conda activate jax-env

# # echo "Running deeponet.py in jax-env..."
# # python3 deeponet.py

# chmod +x run.sh
# ./run.sh


# echo "All done!"
#!/bin/bash

# Exit on error
set -e

# echo "=== Running train_deeponet.py ==="
# python3 train_deeponet.py

echo "=== Running try.py ==="
python3 try.py

echo "=== Running train_deeponet_taylor.py ==="
python3 train_deeponet_taylor.py

echo "=== Running train_deeponet_taylor_split.py ==="
python3 train_deeponet_taylor_split.py

echo "=== All scripts finished successfully ==="
