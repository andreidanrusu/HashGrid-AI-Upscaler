import open3d as o3d

# Load the mesh
mesh_path = "C:/Users/andre/PyCharmProjects/PC3D/data/nerf_synthetic/lego/base.obj"
mesh = o3d.io.read_triangle_mesh(mesh_path)

# Check if the mesh is loaded correctly
if not mesh.has_vertices():
    print("Error: Mesh file not found or empty!")

# Compute vertex normals for better visualization
mesh.compute_vertex_normals()

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])
