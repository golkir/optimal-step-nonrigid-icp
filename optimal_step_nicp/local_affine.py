import torch
import torch.nn as nn


class AffineTransformLocal(nn.Module):
    """
    Implements a local affine transformation module as a neural network layer. This class is designed
    to apply affine transformations to features or coordinates in a localized manner, allowing for
    different transformations at different spatial locations or feature positions.
    The module includes stiffness term to ensure that the close points have similar transformation.
    """

    def __init__(self, num_points, batch_size=1, edges=None):
        """
        Initializes the LocalAffine module with the specified number of points and batch size.
        """
        super(AffineTransformLocal, self).__init__()
        self.A = nn.Parameter(
            torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(
                batch_size, num_points, 1, 1))  # B * N * 3 * 3
        self.b = nn.Parameter(
            torch.zeros(3).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(
                batch_size, num_points, 1, 1))  #B * N * 3 * 1
        self.edges = edges
        self.num_points = num_points

    def stiffness(self):
        if self.edges is None:
            raise Exception("No edges provided")
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        affine_weight = torch.cat((self.A, self.b), dim=3)  # B * N * 3 * 4
        w1 = torch.index_select(affine_weight, dim=1, index=idx1)
        w2 = torch.index_select(affine_weight, dim=1, index=idx2)
        w_diff = (w1 - w2)**2
        return w_diff

    def forward(self, x, pool_num=0, return_stiff=False):
        '''
            x should have shape of B * N * 3
        '''
        x = x.unsqueeze(3)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b
        out_x.squeeze_(3)
        if return_stiff:
            stiffness = self.stiffness()
            return out_x, stiffness, self.A
        else:
            return out_x


if __name__ == "__main__":

    # Test the LocalAffine module
    num_points = 10
    batch_size = 2
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                          [6, 7], [7, 8], [8, 9], [9, 0]])
    x = torch.randn(batch_size, num_points, 3)
    local_affine = AffineTransformLocal(num_points, batch_size, edges)
    out_x, stiffness, A = local_affine(x, return_stiff=True)
    print(out_x.shape, stiffness.shape, A.shape)
    print(out_x)
    print(stiffness)
    print(A)
