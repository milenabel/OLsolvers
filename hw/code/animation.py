import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define file path
file_1_path = "../data/cleanExo_1.mat"

figs_dir = "../figs"
os.makedirs(figs_dir, exist_ok=True)

# Load data
with h5py.File(file_1_path, 'r') as f1:
    time_steps = f1['time'][()].flatten()  # Extract time vector
    x_positions = f1['nvar01'][()]  # Shape (20, 8)
    y_positions = f1['nvar02'][()]  # Shape (20, 8)
    z_positions = f1['nvar03'][()]  # Shape (20, 8)

    print(f"Time steps: {time_steps.shape[0]} (First 5 values: {time_steps[:5]})")
    print(f"X positions shape: {x_positions.shape}, Y positions shape: {y_positions.shape}, Z positions shape: {z_positions.shape}")

# Create animation figure
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(x_positions[0, :], y_positions[0, :], c='blue', s=50)

# Set axis limits based on data range
ax.set_xlim(np.min(x_positions) - 0.05, np.max(x_positions) + 0.05)
ax.set_ylim(np.min(y_positions) - 0.05, np.max(y_positions) + 0.05)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Point Movement Over Time")

# Animation function
def update(frame):
    sc.set_offsets(np.c_[x_positions[frame, :], y_positions[frame, :]])
    ax.set_title(f"Point Movement Over Time\nTime Step: {frame+1}/{len(time_steps)}")
    return sc,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=200, blit=False)

# Save animation as a GIF
gif_path = os.path.join(figs_dir, "p1_point_movement.gif")
ani.save(gif_path, writer="pillow", fps=5)
print(f"Animation saved at: {gif_path}")

# Display animation
plt.show()


from PIL import Image
import os

gif_path = "../figs/p1_point_movement.gif"
output_dir = "../figs/frames"

os.makedirs(output_dir, exist_ok=True)

img = Image.open(gif_path)
for i in range(img.n_frames):
    img.seek(i)
    img.save(os.path.join(output_dir, f"p1_point_movement_{i+1}.png"))

print("Frames extracted successfully!")
