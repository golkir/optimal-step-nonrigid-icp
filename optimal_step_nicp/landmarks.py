import os
import torch
import open3d as o3d
import numpy as np
import trimesh
import cv2
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python

# local imports
from optimal_step_nicp.draw import visualize_landmarks
from optimal_step_nicp import DATADIR, MODELSDIR
from optimal_step_nicp.render import render_mesh_as_image


def _prepare_mesh(mesh):
    mesh_verts = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.triangles)
    mesh.textures = o3d.utility.Vector3dVector(
        [])  # Set textures to an empty list
    mesh.triangle_uvs = o3d.utility.Vector2dVector([])
    mesh.compute_vertex_normals()

    # Create a new mesh with the same geometry but with per-vertex colors
    shape_mesh = o3d.geometry.TriangleMesh()
    shape_mesh.vertices = o3d.utility.Vector3dVector(mesh_verts)
    shape_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
    # shape_mesh.compute_vertex_normals()
    shape_mesh.vertex_colors = o3d.utility.Vector3dVector(
        mesh_verts
    )  # assign color based on verts position to retrieve landmarks later

    return mesh, shape_mesh


def _gather_landmark_vertices(landmarks, shape_img, device):
    # select row element (corresponding to hz position in image) of landmarks
    row_index = landmarks[:, :, 1].view(landmarks.shape[0], -1)
    # select column element (corresponding to vertical position in image) of landmarks
    column_index = landmarks[:, :, 0].view(landmarks.shape[0], -1)
    row_index = row_index.unsqueeze(2).unsqueeze(3).expand(
        landmarks.shape[0], landmarks.shape[1], shape_img.shape[2],
        shape_img.shape[3])
    column_index = column_index.unsqueeze(1).unsqueeze(3).expand(
        landmarks.shape[0], landmarks.shape[1], landmarks.shape[1],
        shape_img.shape[3])

    shape_img = torch.from_numpy(shape_img).to(device).float()
    lm_vertices = torch.gather(shape_img, 1, row_index)
    lm_vertices = torch.gather(lm_vertices, 2, column_index)
    lm_vertices = torch.diagonal(lm_vertices, dim1=1, dim2=2)
    lm_vertices = lm_vertices.transpose(1, 2).float()
    return lm_vertices


def _adjust_face_landmarks(
    on_surface_mask,
    lm_vertices,
):
    """
    # measure whether the lip points locate on the surfaces
    # outer lip is supposed to locate on the surface
    # inner lip is possible to locate on the mouth interior, we shall remove these points during registration
    """

    outer_lip = lm_vertices[:, 48:61]
    inner_lip = lm_vertices[:, 61:]
    lip_threshold = outer_lip[:, 6, 0] - outer_lip[:, 0, 0]
    outer_lip_z = torch.mean(outer_lip[:, :, 2], dim=1, keepdim=True)
    inner_lip_z = torch.abs(inner_lip[:, :, 2] - outer_lip_z)
    inner_lip_mask = inner_lip_z < lip_threshold.unsqueeze(1)

    on_surface_mask[:, 61:] = torch.logical_and(on_surface_mask[:, 61:],
                                                inner_lip_mask)
    return on_surface_mask


def detectLMMediapipe(
    model_file,
    img: np.ndarray,
):
    base_options = python.BaseOptions(model_asset_path=model_file)
    options = vision.PoseLandmarkerOptions(base_options=base_options,
                                           output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    # transform to mediapipe format
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(img)
    landmarks = detection_result.pose_landmarks
    return landmarks


def get_pose_landmarks(mesh, device, lm_detector="mediapipe", visualize=False):
    """
    Extracts landmark points from a given mesh. This function identifies specific points on the mesh
    that correspond to predefined landmark positions, such as the tip of the nose, corners of the eyes,
    and mouth, etc. These landmarks are crucial for various tasks like facial alignment, animation, and
    morphing.

    Parameters:
    - mesh (o3d.geometry.TriangleMesh or similar): The mesh from which landmarks are to be extracted.
      The mesh should have vertices and possibly textures defined.
    - device (str, optional): The computing device ('cpu' or 'cuda') where the operation will be performed.
      Defaults to 'cpu'.

    Returns:
    - torch.Tensor: A tensor of shape (N, 3) containing the 3D coordinates of the N extracted landmark points.
      The tensor is located on the specified device.
    """

    model_file = os.path.join(MODELSDIR, "pose_landmarker.task")
    mesh, shape_mesh = _prepare_mesh(mesh)

    # render shape mesh
    shape_img = render_mesh_as_image(shape_mesh)
    shape_img = np.expand_dims(shape_img, axis=0)

    # render target mesh
    img = render_mesh_as_image(mesh)
    img = (img * 255).astype(np.uint8)  # should be in 0-255 range

    img_uint8 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_uint8 = np.transpose(img, (2, 0, 1))
    img_uint8 = torch.from_numpy(img_uint8).unsqueeze(0).to(device)
    img_uint8_np = img_uint8.cpu().numpy()[0].transpose(1, 2, 0)

    x_shape, y_shape = img_uint8_np.shape[1], img_uint8_np.shape[0]

    if lm_detector == "mediapipe":
        landmarks = detectLMMediapipe(model_file, img_uint8_np)
        landmarks = np.asarray([[l.x, l.y, l.z] for l in landmarks[0]],
                               dtype=np.float32).astype(np.float32)  # 33, 3
        landmarks = torch.from_numpy(landmarks).to(
            device)  # range between [0, 1]
        # need to de-normalize the landmarks because they are in [0, 1] range now
        landmarks[:, 0] = landmarks[:, 0] * x_shape
        landmarks[:, 1] = landmarks[:, 1] * y_shape
        landmarks = landmarks.long().unsqueeze(0)

    lm_vertices = _gather_landmark_vertices(landmarks, shape_img, device)

    # Compute the pairwise distance matrix
    mesh_verts = torch.from_numpy(np.asarray(mesh.vertices)).to(device).float()
    distances = torch.cdist(lm_vertices.squeeze(0), mesh_verts)
    # Get the indices of the nearest neighbor for each point
    _, indices = torch.topk(distances, 1, largest=False)
    lm_index = indices.squeeze(-1)
    if visualize:
        visualize_landmarks(mesh, lm_index.unsqueeze(1))

    return lm_index.unsqueeze(0).long()


if __name__ == "__main__":

    target_mash_path = os.path.join(DATADIR, "target.ply")

    template_mesh_path = os.path.join(DATADIR, "template.obj")

    target_point_cloud = o3d.io.read_triangle_mesh(target_mash_path)
    tri_mesh = trimesh.load_mesh(template_mesh_path, process=True)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    lm_indices = get_pose_landmarks(o3d_mesh, device="cpu", visualize=True)

    print("Landmarks extracted successfully")
