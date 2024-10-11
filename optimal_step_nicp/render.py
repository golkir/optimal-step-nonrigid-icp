import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import open3d.visualization.rendering as rendering
from optimal_step_nicp.utils import normalize_mesh
from optimal_step_nicp import DATADIR


def render_mesh_as_image(mesh, image_size=1000, camera_direction=None):
    """
    Renders the mesh using Open3D without applying any lighting model.
    
    Args:
    - mesh (o3d.geometry.TriangleMesh): The mesh to render.
    - camera_direction (tuple): The camera direction as (eye, center, up).
    - image_size (int): The size of the output image.
    
    Returns:
    - np.array: The rendered image as a NumPy array.
    """

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, height=image_size, width=image_size)
    vis.add_geometry(mesh)
    # vis.get_render_option().light_on = True

    vis.poll_events()
    vis.update_renderer()

    # Render the image

    img = vis.capture_screen_float_buffer(do_render=True)
    img = np.asarray(img)
    vis.capture_screen_image(os.path.join(DATADIR, "output.png"),
                             do_render=True)
    vis.destroy_window()
    return img


if (__name__ == "__main__"):

    mesh_path = os.path.join(DATADIR, "template.obj")

    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    mesh, _ = normalize_mesh(mesh)
    mesh.compute_vertex_normals()

    camera_direction = (
        np.asarray([0, 0, 0]), np.asarray([1, 0, 0]), np.asarray([0, 0, 1])
    )  # Eye to the right, looking at the center, with up direction

    img = render_mesh_as_image(mesh, camera_direction=camera_direction)
    plt.imshow(img)
