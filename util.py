##
import torch
import random

#input : 3 coordinates Per N Points, distance K
#output : kNN distance of N points (NxK) kNN indicies of N Points NxK
def find_knn(point_cloud, k):
    N = len(point_cloud)
    delta = point_cloud.view(N, 1, 3) - point_cloud.view(1, N, 3) #NxNx3이됨 by broadcasting
    dist = torch.sum(delta**2, dim=-1) #NxN

    knn_dist, knn_indicies = dist.topk(k=k, dim=-1, largest=False)

    return knn_dist, knn_indicies
##

#when query set and point set are not same
#input : dataset point Nx3, Query Point Mx3, K
#output : kNN distance MxK, kNN indicies MxK
def find_knn_general(query_points, dataset_points, k):
    M = len(query_points)
    N = len(dataset_points)

    delta = query_points.view(M, 1, 3) - dataset_points.view(1, N, 3) #MxNx3
    dist = torch.sum(delta**2, dim=-1) #MxN

    knn_dist, knn_indicies = dist.topk(k=k, dim=-1, largest=False) #M,K

    return knn_dist, knn_indicies
##
#input : dataset (Nx3), corresponding feature (NxC), query (Mx3), K
def interpolate_knn(query_points, dataset_points, dataset_features, k):
    M = len(query_points)
    N, C = dataset_features.shape

    knn_dist, knn_indices = find_knn_general(query_points, dataset_points, k)
    knn_dataset_features = dataset_features[knn_indices.view(-1)].view(M, k, C) #->(M*KxC)->MxKxC

    #calculate interpolation weights
    knn_dist_recip = 1. / (knn_dist + 1e-8) #MxK
    denom = knn_dist_recip.sum(dim=-1, keepdim=True) #Mx1
    weights = knn_dist_recip / denom #MxK

    #Linear Interpolation
    weighted_features = weights.view(M,k,1) * knn_dataset_features #MxKxC
    interpolated_features = weighted_features.sum(dim=1) #MxC

    return interpolated_features
##
#input : point Nx3, number of sample M
#output : sampled_indicies (M,)
def fps(points, num_samples):
    N = len(points)
    sampled_indicies = torch.zeros(num_samples, dtype=torch.long) #init
    distance = torch.ones(N,) * 1e10
    farthest_idx = random.randint(0, N)

    for i in range(num_samples):
        #sample farthest point
        sampled_indicies[i] = farthest_idx
        centroid = points[farthest_idx].view(1,3)
        #compute distance
        delta = points - centroid
        dist = torch.sum(delta**2, dim=-1) #N,

        mask = dist < distance #중복계산을 피하기 위해 -> fps는 가장 먼 거리부터 샘플링하므로 남은 점들은 현재 가장 먼 점보다 가까울 수 밖에 없다. 하나씩 줄어드는 것
        distance[mask] = dist[mask]
        #sample the next farthest
        farthest_idx = torch.max(distance, -1)[1] #maximum 값의 index

    return sampled_indicies

##
if __name__  == '__main__':
    M = 5
    N = 20
    K = 3
    query_points = torch.randn(M, 3)
    dataset_points = torch.randn(N, 3)
    knn_dist, knn_indices = find_knn(query_points, K)
    knn_dist2, knn_indices2 = find_knn_general(query_points, dataset_points, K)
    print(knn_dist.shape)
    print(knn_indices.shape)
    print(knn_dist2.shape)
    print(knn_indices2.shape)
    print(knn_dist2)

    C = 16
    dataset_features = torch.randn(N,C)

    interpolateed_features = interpolate_knn(query_points, dataset_points, dataset_features, K)
    print(interpolateed_features.shape)

    points = torch.randn(N,3)
    sampled_indicies = fps(points, M)
    print(sampled_indicies.shape)
    sampled_points = points[sampled_indicies]
    print(sampled_points.shape)
##

