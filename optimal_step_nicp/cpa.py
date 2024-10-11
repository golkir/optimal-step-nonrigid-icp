import torch


def corresponding_points_alignment(source_points, target_points):
    """
    Based on [1] Shinji Umeyama: Least-Suqares Estimation of Transformation Parameters Between Two Point Patterns
    """

    assert source_points.shape == target_points.shape
    assert len(source_points.shape) == 3
    assert source_points.shape[2] == 3

    # Compute centroids
    centroid_source = torch.mean(source_points, dim=1, keepdim=True)
    centroid_target = torch.mean(target_points, dim=1, keepdim=True)

    # Center the points
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # Compute the covariance matrix
    H = torch.bmm(source_centered.transpose(1, 2), target_centered)

    # Compute the Singular Value Decomposition (SVD)
    U, S, V = torch.svd(H)

    # Compute the rotation matrix
    R = torch.bmm(V, U.transpose(1, 2))

    # Correct for reflection case
    det = torch.det(R)
    V[det < 0, :, -1] *= -1
    R = torch.bmm(V, U.transpose(1, 2))

    # Compute the translation vector

    t = centroid_target - torch.bmm(centroid_source, R)

    # Compute the scaling factor
    scaling = torch.sum(S, dim=1) / torch.sum(source_centered**2, dim=[1, 2])

    return R, t, scaling
