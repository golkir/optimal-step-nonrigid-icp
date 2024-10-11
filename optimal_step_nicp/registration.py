import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# local imports
from optimal_step_nicp.draw import visualize_landmarks, draw_registration_result
from optimal_step_nicp.local_affine import AffineTransformLocal
import optimal_step_nicp.utils as utils
from optimal_step_nicp.utils import convert_mesh_to_pcl, mesh_boundary, laplacian_smoothing, batch_vertex_sample
from optimal_step_nicp.cpa import corresponding_points_alignment


def registration_mesh2mesh(template_mesh: o3d.geometry.TriangleMesh,
                           target_mesh: o3d.geometry.TriangleMesh,
                           template_lm_index: torch.LongTensor,
                           target_lm_index: torch.LongTensor,
                           config: dict,
                           device=torch.device('cpu')):
    target_pcl = convert_mesh_to_pcl(target_mesh)
    return registration_mesh2pcl(template_mesh,
                                 target_pcl,
                                 template_lm_index,
                                 target_lm_index,
                                 config,
                                 device=device)


def registration_mesh2pcl(template_mesh: o3d.geometry.TriangleMesh,
                          target_pcl: o3d.geometry.PointCloud,
                          template_lm_index: torch.LongTensor,
                          target_lm_index: torch.LongTensor,
                          config: dict,
                          out_affine=False,
                          in_affine=None,
                          device=torch.device('cpu')):
    """
    Performs non-rigid iterative closest point (ICP) algorithm to align a source mesh to a target mesh
    using a set of corresponding landmarks. This function iteratively refines the alignment by minimizing
    the distance between the source and target meshes under the constraint of landmark correspondences.

    Parameters:
    - template_mesh (o3d.geometry.TriangleMesh): The source mesh to be aligned to the target mesh.
    - target_mesh (o3d.geometry.TriangleMesh): The target mesh to which the source mesh is aligned.
    - template_landmarks (torch.Tensor): A tensor of shape (N, 3) containing N 3D landmark points on the source mesh.
    - target_landmarks (torch.Tensor): A tensor of shape (N, 3) containing N 3D landmark points on the target mesh.
      The landmarks in source_landmarks correspond to those in target_landmarks.
    - config (dict): A dictionary containing configuration parameters for the non-rigid ICP algorithm, such as
      the number of iterations, regularization parameters, and convergence criteria.
    - out_affine (bool): A flag indicating whether to return the learned local affine transformation model.
    - in_affine (torch.nn.Module): A pre-trained local affine transformation model.
    - device (torch.device): The device on which to perform computations (e.g., 'cpu' or 'cuda').

    Returns:
    - o3d.geometry.TriangleMesh: The aligned source mesh after applying the non-rigid ICP algorithm.
    """
    # Config
    inner_iter = config['inner_iter']
    outer_iter = config['outer_iter']
    log_iter = config['log_iter']
    loop = tqdm(range(outer_iter))

    milestones = set(config['milestones'])
    stiffness_weights = np.array(config['stiffness_weights'])
    landmark_weights = np.array(config['landmark_weights'])
    laplacian_weight = config['laplacian_weight']
    w_idx = 0

    # Tmeplate vertices and faces

    template_vertex = torch.tensor(np.asarray(
        template_mesh.vertices)).float().unsqueeze(0)
    template_faces = torch.tensor(np.asarray(
        template_mesh.triangles)).long().unsqueeze(0)

    # target vertices
    target_vertex = torch.tensor(np.asarray(
        target_pcl.points)).float().unsqueeze(0)

    assert target_vertex.shape[0] == 1  # batch size should be 1

    # mesh boundary mask
    boundary_mask = mesh_boundary(template_faces[0], template_vertex.shape[1])
    boundary_mask = boundary_mask.unsqueeze(0).unsqueeze(2)

    # mesh interior mask
    inner_mask = torch.logical_not(boundary_mask)

    # remove landmarks which are far apart

    target_lm = batch_vertex_sample(target_lm_index, target_vertex)
    template_lm = batch_vertex_sample(template_lm_index, template_vertex)

    distances = torch.norm(target_lm - template_lm, dim=2)
    target_lm_index = target_lm_index[distances < 0.5].unsqueeze(0)
    template_lm_index = template_lm_index[distances < 0.5].unsqueeze(0)

    target_lm = target_lm[distances < 0.5].unsqueeze(0)
    template_lm = template_lm[distances < 0.5].unsqueeze(0)

    # rigid alignment of landmarks to get the initial transformation
    # SVD-based alignment https://www.youtube.com/watch?v=dhzLQfDBx2Q

    R, T, s = corresponding_points_alignment(target_lm, template_lm)

    # transform the template vertices based on the rigid alignment
    transformed_vertex = s * torch.bmm(template_vertex, R) + T

    trans_mesh = utils.create_mesh_from_mesh(template_mesh,
                                             transformed_vertex[0])

    target_vertex_np = np.asarray(target_pcl.points)

    # build a KDTree for efficient nearest neighbor search
    tree = KDTree(target_vertex_np)

    # define the transformation model (Local Affine Transform)
    triangles = np.array(template_mesh.triangles)

    edges = np.vstack(
        [triangles[:, :2], triangles[:, 1:3], triangles[:, [0, 2]]])
    edges = np.sort(edges, axis=1)  # sort the vertices for each edge
    edges = np.unique(edges, axis=0)  # remove duplicate edges

    template_edges = torch.tensor(edges, dtype=torch.long).to(device)

    if in_affine is None:
        local_affine_model = AffineTransformLocal(template_vertex.shape[1],
                                                  template_vertex.shape[0],
                                                  template_edges).to(device)
    else:
        local_affine_model = in_affine

    # define optimizer

    optimizer = torch.optim.AdamW([{
        'params': local_affine_model.parameters()
    }],
                                  lr=1e-4,
                                  amsgrad=True)

    for i in loop:
        # just uses linear transformation based on learned parameters and also uses stiffness term
        new_deformed_verts, stiffness, mesh_transformation = local_affine_model(
            transformed_vertex, pool_num=0, return_stiff=True)

        # newly deformed landmarks
        new_deformed_lm = batch_vertex_sample(template_lm_index,
                                              new_deformed_verts)

        old_verts = new_deformed_verts
        new_deformed_mesh = template_mesh

        # set new template vertices based on transformation

        new_deformed_mesh.vertices = o3d.utility.Vector3dVector(
            new_deformed_verts.squeeze(0).detach().cpu().numpy().astype(
                np.float64))

        # we can randomly sample the target point cloud for speed up

        # Convert tensors to numpy arrays

        new_deformed_verts_np = new_deformed_verts.squeeze(
            0).detach().cpu().numpy()

        # Query the KDTree for the nearest neighbor to find closeset points on target mesh/point cloud
        distances, indices = tree.query(new_deformed_verts_np, k=1)

        indices = torch.from_numpy(indices).to(device)
        close_points = target_vertex[0, indices, :]

        if (i == 0) and (in_affine is None):
            inner_loop = range(4)
        else:
            inner_loop = range(inner_iter)

        # enter inner loop
        for _ in inner_loop:
            optimizer.zero_grad()

            vert_distance = (new_deformed_verts - close_points)**2
            bsize = vert_distance.shape[0]

            # we need a sum over vector components for L2 norm. Set 0.04 as threshold
            vert_distance_mask = torch.sum(vert_distance, dim=2) < 0.04**2

            # Logical AND with inner mask to remove boundary vertices
            weight_mask = torch.logical_and(inner_mask,
                                            vert_distance_mask.unsqueeze(2))

            # multipley mask by vert_distance to select vertex that match conditions,
            # specifically that the distance should be less than 0.04**2
            # multiplying False/True by number gives 0/1

            vert_distance = weight_mask * vert_distance
            vert_distance = vert_distance.view(bsize, -1)

            # This is the first term of the Loss function
            vert_sum = torch.sum(vert_distance) / bsize

            # distance between deformed template landmarks and target landmarks
            landmark_distance = (new_deformed_lm - target_lm)**2

            bsize = vert_distance.shape[0]

            landmark_distance = landmark_distance.view(bsize, -1)

            # Loss term. L2 loss for landmark distance weighted by landmark weights
            landmark_sum = torch.sum(
                landmark_distance) * landmark_weights[w_idx] / bsize

            # Stiffness loss  term. L2 loss for stiffness weighted by stiffness weights
            stiffness = stiffness.view(bsize, -1)
            stiffness_sum = torch.sum(
                stiffness) * stiffness_weights[w_idx] / bsize

            # Laplacian smoothing loss term
            # It describes how a vertex deviates from the average of its neighbors
            laplacian_loss = laplacian_smoothing(new_deformed_mesh)

            # Laplacian weight
            laplacian_loss = laplacian_loss * laplacian_weight

            # sum up all the loss terms
            loss = torch.sqrt(vert_sum + landmark_sum +
                              stiffness_sum) + laplacian_loss
            loss.backward()
            optimizer.step()
            # here we again use transformed_vertex (obtained as a result of initial rigid transformation
            # of landmarks).

            new_deformed_verts, stiffness, mesh_transformation = local_affine_model(
                transformed_vertex, pool_num=0, return_stiff=True)
            new_deformed_lm = batch_vertex_sample(template_lm_index,
                                                  new_deformed_verts)

            template_mesh.vertices = o3d.utility.Vector3dVector(
                new_deformed_verts.squeeze(0).detach().cpu().numpy().astype(
                    np.float64))
            new_deformed_mesh = template_mesh

        # final loss calculation in outer loop

        distance = torch.mean(
            torch.sqrt(torch.sum((old_verts - new_deformed_verts)**2, dim=2)))
        if i % log_iter == 0:
            print(distance, stiffness_sum.item(), landmark_sum.item(),
                  vert_sum.item(), laplacian_loss.item())

        if i in milestones:
            w_idx += 1

    template_mesh.vertices = o3d.utility.Vector3dVector(
        new_deformed_verts.squeeze(0).detach().cpu().numpy().astype(
            np.float64))
    new_deformed_mesh = template_mesh
    if out_affine:
        return new_deformed_mesh, local_affine_model
    else:
        return new_deformed_mesh
