import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d


def normalize_mesh(in_mesh: o3d.geometry.TriangleMesh):
    in_mesh.remove_duplicated_vertices()
    in_mesh.remove_duplicated_triangles()
    in_mesh.remove_non_manifold_edges()
    # Compute the axis-aligned bounding box (AABB) of the mesh
    vertices_torch = torch.tensor(np.asarray(in_mesh.vertices)).float()
    aabb = in_mesh.get_axis_aligned_bounding_box()
    # Calculate the scale of the mesh
    distance = np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound)
    max_distance = np.max(distance)

    # Calculate the center of the AABB
    center = torch.mean(vertices_torch, dim=0, keepdim=False).cpu().numpy()

    # Calculate the offset to centralize the mesh
    offset = -1 * center

    # Calculate the scale factor to normalize the mesh size
    scale = 1 / max_distance

    # Apply the translation to centralize the mesh
    in_mesh.translate(offset, relative=True)

    # Scale the mesh to normalize its size
    in_mesh.scale(scale, center=(0, 0, 0))

    # Return the modified mesh along with the inverse transformation parameters
    inverse_transform = (-offset, 1 / scale)
    return in_mesh, inverse_transform


# converted
def normalize_pcl(pcl: o3d.geometry.PointCloud):
    '''
        Normalize a PointCloud by translating it to the origin and scaling it to fit within a unit cube.

        input: PointCloud object
        return: Normalized PointCloud object
    '''
    # Convert to numpy array
    points = np.asarray(pcl.points)

    # Translate to the origin
    centroid = points.mean(axis=0)
    points -= centroid

    # Scale to fit within a unit cube
    max_range = points.ptp(axis=0).max()
    points /= max_range

    # Create a new PointCloud object
    pcl_normalized = o3d.geometry.PointCloud()
    pcl_normalized.points = o3d.utility.Vector3dVector(points)

    return pcl_normalized


def pointcloud_normal(in_pcl: o3d.geometry.PointCloud):
    '''
        o3d normal estimation
    '''
    in_pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    return torch.from_numpy(np.asarray(in_pcl.normals))


def mesh_boundary(in_faces: torch.LongTensor, num_verts: int):
    '''
    input:
        in edges: N * 3, is the vertex index of each face, where N is number of faces
        num_verts: the number of vertexs mesh
    return:
        boundary_mask: bool tensor of num_verts, if true, point is on the boundary, else not
    '''
    in_x = in_faces[:, 0]
    in_y = in_faces[:, 1]
    in_z = in_faces[:, 2]
    in_xy = in_x * (num_verts) + in_y
    in_yx = in_y * (num_verts) + in_x
    in_xz = in_x * (num_verts) + in_z
    in_zx = in_z * (num_verts) + in_x
    in_yz = in_y * (num_verts) + in_z
    in_zy = in_z * (num_verts) + in_y
    in_xy_hash = torch.minimum(in_xy, in_yx)
    in_xz_hash = torch.minimum(in_xz, in_zx)
    in_yz_hash = torch.minimum(in_yz, in_zy)
    in_hash = torch.cat((in_xy_hash, in_xz_hash, in_yz_hash), dim=0)
    output, count = torch.unique(in_hash, return_counts=True, dim=0)
    boundary_edge = output[count == 1]
    boundary_vert1 = boundary_edge // num_verts
    boundary_vert2 = boundary_edge % num_verts
    boundary_mask = torch.zeros(num_verts).bool()
    boundary_mask[boundary_vert1] = True
    boundary_mask[boundary_vert2] = True
    return boundary_mask


def convert_mesh_to_pcl(in_mesh: o3d.geometry.TriangleMesh):
    '''
        Convert Meshes object to Pointclouds object(only converting vertexes)
        input: TriangleMesh object, number of points to sample
        return: PointCloud object
    '''
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(np.asarray(in_mesh.vertices))
    return pcl


# not sure
def batch_vertex_sample(batch_idx: torch.LongTensor, vertex: torch.Tensor):
    '''
    input:
        batch_idx: shape of (B * L), B is the batch size, L is the select point length
        vertex: shape of (B * N * 3), N is the vertex size
    output:
        vertex: (B * L * 3)
    '''
    batch_idx_expand = batch_idx.unsqueeze(2).expand(batch_idx.shape[0],
                                                     batch_idx.shape[1],
                                                     vertex.shape[2])
    sampled_vertex = torch.gather(vertex, 1, batch_idx_expand)
    return sampled_vertex


def get_adjacency_matrix(mesh: o3d.geometry.TriangleMesh):
    """
    input:
        mesh: open3d TriangleMesh object
    output:
        adjacency_matrix: sparse matrix of shape (num_verts, num_verts)
        degree_matrix: sparse matrix of shape (num_verts, num_verts)
    """
    # Get the triangles of the mesh
    triangles = np.array(mesh.triangles)  # Convert to numpy array first
    triangles = torch.from_numpy(triangles)  # Then convert to tensor

    # Create row and column indices for the adjacency matrix
    row_indices = triangles[:, [0, 1, 2, 0, 1]].flatten()
    col_indices = triangles[:, [1, 2, 0, 2, 0]].flatten()

    # Create the adjacency matrix as a sparse tensor
    adjacency_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=torch.ones(len(row_indices)),
        size=(len(mesh.vertices), len(mesh.vertices)))

    # Compute degree values
    degree_values = torch.sparse.sum(adjacency_matrix, dim=1).to_dense()

    # Create the degree matrix as a sparse tensor
    degree_matrix = torch.sparse_coo_tensor(indices=torch.stack([
        torch.arange(degree_values.shape[0]),
        torch.arange(degree_values.shape[0])
    ]),
                                            values=degree_values,
                                            size=(degree_values.shape[0],
                                                  degree_values.shape[0]))

    return adjacency_matrix, degree_matrix


def laplacian_smoothing(mesh: o3d.geometry.TriangleMesh, lamb=0.5):
    # Get adjacency matrix
    adjacency_matrix, degree_matrix = get_adjacency_matrix(mesh)

    # Convert mesh vertices to PyTorch tensor
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)

    # # Calculate degree matrix
    # degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))

    # Calculate Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Calculate Laplacian of each vertex
    laplacian = torch.matmul(laplacian_matrix, vertices)

    # Calculate Laplacian loss
    laplacian_loss = torch.sum(laplacian**2)

    # Move each vertex towards the average position of its neighbors
    new_vertices = vertices - lamb * laplacian

    # Update mesh vertices
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices.detach().numpy())

    return laplacian_loss


def knn_points(point_cloud, query_points, k=1):
    """
    Compute KNN for a point cloud and a set of query points.

    Parameters:
    - point_cloud: Tensor of shape (N, D) where N is the number of points in the point cloud, and D is the dimension.
    - query_points: Tensor of shape (M, D) where M is the number of query points.
    - k: The number of nearest neighbors to find.

    Returns:
    - distances: Tensor of shape (M, k) containing the distances of the k nearest neighbors for each query point.
    - indices: Tensor of shape (M, k) containing the indices of the k nearest neighbors for each query point.
    """
    # Compute pairwise distance matrix
    diff = point_cloud.unsqueeze(1) - query_points.unsqueeze(2)
    dist_squared = torch.sum(diff**2, dim=-1)

    # Find the k nearest neighbors
    distances, indices = torch.topk(dist_squared,
                                    k=k,
                                    largest=False,
                                    sorted=True)

    return distances, indices


def rotate(template_vertex):
    # Define a rotation matrix for 180 degrees rotation around the X-axis
    theta = np.pi  # 180 degrees in radians
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [
                                        0,
                                        torch.cos(torch.tensor([theta])),
                                        -torch.sin(torch.tensor([theta]))
                                    ],
                                    [
                                        0,
                                        torch.sin(torch.tensor([theta])),
                                        torch.cos(torch.tensor([theta]))
                                    ]])

    # Apply the rotation to the template vertices
    template_vertex_rotated = torch.bmm(template_vertex,
                                        rotation_matrix.unsqueeze(0))
    return template_vertex_rotated


def create_mesh_from_mesh(basemesh, newvertices: torch.Tensor):
    newmesh = o3d.geometry.TriangleMesh()
    newmesh.vertices = o3d.utility.Vector3dVector(newvertices.cpu().numpy())
    newmesh.triangles = basemesh.triangles
    newmesh.textures = basemesh.textures
    newmesh.compute_vertex_normals()
    return newmesh
