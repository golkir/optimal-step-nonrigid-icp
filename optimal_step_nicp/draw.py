import copy

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def visualize_landmarks(mesh, lm_indices, isIndexes=False, landmarks=None):
    mesh_verts_np = np.asarray(mesh.vertices)
    landmarks_pcd = o3d.geometry.PointCloud()
    if not isIndexes:
        landmarks_pcd.points = o3d.utility.Vector3dVector(
            mesh_verts_np[lm_indices].squeeze(1))
    else:
        landmarks = np.asarray(landmarks.squeeze().detach().cpu())
        landmarks_pcd.points = o3d.utility.Vector3dVector(landmarks)

    mesh.paint_uniform_color([0.5, 0.5, 0.5])  # gray
    landmarks_pcd.paint_uniform_color([0, 0, 1])  # blue
    print("visualizing landmarks")
    o3d.visualization.draw_geometries([landmarks_pcd, mesh])


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draws pose landmarks on the input image.

    Args:
        rgb_image (numpy.ndarray): The input RGB image.
        detection_result (mediapipe.framework.formats.detection_result_pb2.DetectionResult): The detection result containing pose landmarks.

    Returns
        numpy.ndarray: The annotated image with pose landmarks.
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx, lms in enumerate(pose_landmarks_list):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x,
                                            y=landmark.y,
                                            z=landmark.z)
            for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image, pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def display_landmarks(im_path, detection_result):
    img = plt.imread(im_path)
    annotated_image = draw_landmarks_on_image(img, detection_result)
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image_bgr)
    plt.axis('off')
    plt.show()
