import os
import json

import torch
import open3d as o3d
import numpy as np
import trimesh

from optimal_step_nicp.utils import normalize_mesh
from optimal_step_nicp.landmarks import get_pose_landmarks
from optimal_step_nicp.registration import registration_mesh2mesh

from optimal_step_nicp import DATADIR


def pose_registration(tmpl_mesh_path: str,
                      target_mesh_path: str,
                      device=torch.device,
                      config_path: str = None):

    # load template mesh
    tri_mesh = trimesh.load_mesh(tmpl_mesh_path, process=True)
    template_mesh = o3d.geometry.TriangleMesh()
    template_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    template_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    template_mesh.compute_vertex_normals()

    # load target mesh
    target_mesh = o3d.io.read_triangle_mesh(target_mesh_path,
                                            enable_post_processing=True)
    target_mesh.compute_vertex_normals()

    with torch.no_grad():
        template_mesh, _ = normalize_mesh(template_mesh)
        target_mesh, _ = normalize_mesh(target_mesh)

        target_lm_index = get_pose_landmarks(target_mesh,
                                             device=device,
                                             visualize=False)
        template_lm_index = get_pose_landmarks(template_mesh,
                                               device=device,
                                               visualize=False)

    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    registered_mesh = registration_mesh2mesh(template_mesh,
                                             target_mesh,
                                             template_lm_index,
                                             target_lm_index,
                                             config,
                                             device=device)

    # Save the mesh to a file
    print(registered_mesh)
    registered_mesh.compute_vertex_normals()
    num_vertices = np.asarray(registered_mesh.vertices).shape[0]
    gray_color = np.ones((num_vertices, 3)) * 0.5  # RGB values for gray
    registered_mesh.vertex_colors = o3d.utility.Vector3dVector(gray_color)

    o3d.io.write_triangle_mesh(os.path.join(DATADIR, "pose-registration.obj"),
                               registered_mesh)


def main():
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    config_path = os.path.join(os.path.dirname(__file__), "config/config.json")
    template_mesh_path = os.path.join(DATADIR, "SMPL_male.obj")
    target_mesh_path = os.path.join(DATADIR, "target.ply")

    pose_registration(template_mesh_path,
                      target_mesh_path,
                      device,
                      config_path=config_path)


if __name__ == "__main__":
    main()
