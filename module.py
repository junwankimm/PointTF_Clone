import torch.nn as nn
from util import *

#input : point Nx3, feature NxCin, K
#output : output feature NxCout
class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.linear_q = nn.Linear(in_channels, out_channels, bias=False)
        self.linear_k = nn.Linear(in_channels, out_channels, bias=False)
        self.linaer_v = nn.Linear(in_channels, out_channels, bias=False)

        self.mlp_attn = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )

        self.mlp_pos = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_channels, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, points, features):
        #Q, K, V
        N = len(points)
        f_q = self.linear_q(features) #NxCout
        f_k = self.linear_k(features)
        f_v = self.linaer_v(features)
        #kNN
        knn_dist, knn_indices = find_knn(points, self.k) #NxK
        knn_points = points[knn_indices.view(-1)].view(N, self.k, 3) #N*Kx1
        knn_k = f_k[knn_indices.view(-1)].view(N, self.k, self.out_channels)
        knn_v = f_v[knn_indices.view(-1)].view(N, self.k, self.out_channels)
        #position
        rel_pos = points.view(N, 1, 3) - knn_points #NxKx3
        rel_pos_enc = self.mlp_pos(rel_pos.view(-1, 3)).view(N, self.k, self.out_channels)#N*Kx3 since we use BN1d -> MLP에 의해 N*Kxoutchannels -> reshape
        #Similarity
        vec_sim = f_q.view(N, 1, self.out_channels) - knn_k + rel_pos_enc
        weights = self.mlp_attn(vec_sim.view(-1, self.out_channels)).view(N, self.k, self.out_channels)
        weights = self.softmax(weights) #softmax across k (dim=1), NxKxCout
        #weighted sum
        weighted_knn_v = weights * (knn_v + rel_pos_enc) #NxKxCout
        out_features = weighted_knn_v.sum(dim=1) #NxCout

        return out_features
##
class PointTransformerBlock(nn.Module):
    def __init__(self, channels, k):
        super().__init__()
        self.linear_in = nn.Linear(channels, channels)
        self.linear_out = nn.Linear(channels, channels)
        self.pt_layer = PointTransformerLayer(channels, channels, k)

    def forward(self, points, features):
        out_features = self.linear_in(features)
        out_features = self.pt_layer(points, out_features)
        out_features = self.linear_out(out_features)
        out_features += features

        return out_features

##
#input : input points Nx3, Features NxC, Sample M, K
#output : smapled Points Mx3, corresponding featrue MxC
class TransitionDown(nn.Module):
    def __init__(self, channels, num_samples, k):
        super().__init__()
        self.k = k
        self.num_samples = num_samples
        self.channels = channels

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, features):
        N = len(points)
        sampled_indicies = fps(points, self.num_samples) #numsamplex1
        sampled_points = points[sampled_indicies] #numsamplex3

        #kNN
        knn_dist, knn_indices = find_knn_general(sampled_points, points, self.k)

        #MLP
        knn_features = features[knn_indices.view(-1)]
        out_knn_features = self.mlp(knn_features).view(self.num_samples, self.k, -1) #MxKxC

        #Local MP
        out_features = out_knn_features.max(dim=1)[0]

        return sampled_points, out_features
##
# input : up_point Nx3, up_features NxC_up, down_points Mx3 down_features MxC_down
# output : out_features NxC_out
class TransitionUp(nn.Module):
    def __init__(self, up_channels, down_channels, out_channels):
        super().__init__()

        self.linear_up = nn.Linear(up_channels, out_channels)
        self.linear_down = nn.Linear(down_channels, out_channels)

    def forward(self, up_points, up_features, down_points, down_features):
        down_f = self.linear_down(down_features)
        interp_f = interpolate_knn(up_points, down_points, down_f, k=3)
        out_f = interp_f + self.linear_up(up_features)

        return out_f

class PointTransformer(nn.Module):
    def __init__(self, in_channels, num_samples, up_channels, down_channels, out_channels, k):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ptf1 = PointTransformerBlock(32, k)
        self.ptf2 = PointTransformerBlock(64, k)
        self.ptf2 = PointTransformerBlock(128, k)
        self.ptf2 = PointTransformerBlock(256, k)
        self.ptf2 = PointTransformerBlock(512, k)
        self.ptf2 = PointTransformerBlock(512, k)
        self.ptf2 = PointTransformerBlock(256, k)
        self.ptf2 = PointTransformerBlock(128, k)
        self.ptf2 = PointTransformerBlock(64, k)
        self.ptf2 = PointTransformerBlock(32, k)

        self.td1 = TransitionDown(32)


##
if __name__ == '__main__':
    N = 20
    C_in = 32
    C_down = 64
    C_out = 128
    M = 5
    K = 7

    points = torch.randn(N, 3)
    features = torch.randn(N, C_in)
    td_module = TransitionDown(C_in, M, K)
    down_points, down_features = td_module(points, features)
    # print(down_points.shape)
    # print(down_features.shape)
    up_points = torch.randn(N,3)
    up_features = torch.randn(N, C_in)
    down_points = torch.randn(M, 3)
    down_features = torch.randn(M, C_down)
    tu_module = TransitionUp(C_in, C_down, C_out)
    out_features = tu_module(up_points, up_features, down_points, down_features)
    print(out_features.shape)

    # pt_layer = PointTransformerLayer(C_in, C_out, K)
    # out_features = pt_layer(points, features)
    # print(out_features.shape)