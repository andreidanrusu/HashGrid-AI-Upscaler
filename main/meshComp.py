import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Function to create a simple high-resolution sphere mesh
def create_mesh(resolution):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    mesh.compute_vertex_normals()
    return mesh

# Generate two meshes with different resolutions
mesh_512 = create_mesh(40)   # Low resolution (512-like)
mesh_2048 = create_mesh(60)  # High resolution (2048-like)

# Visualizing the two meshes side by side
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)

# Capture low-res mesh
vis.add_geometry(mesh_512)
vis.poll_events()
vis.update_renderer()
image_512 = vis.capture_screen_float_buffer(True)
vis.clear_geometries()

# Capture high-res mesh
vis.add_geometry(mesh_2048)
vis.poll_events()
vis.update_renderer()
image_2048 = vis.capture_screen_float_buffer(True)

# Close visualizer
vis.destroy_window()

# Convert Open3D images to numpy arrays and plot them
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(np.asarray(image_512))
ax[0].axis("off")
ax[0].set_title("512³ Resolution Mesh")

ax[1].imshow(np.asarray(image_2048))
ax[1].axis("off")
ax[1].set_title("2048³ Resolution Mesh")

plt.show()
