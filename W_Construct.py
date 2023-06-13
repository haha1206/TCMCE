from sklearn.metrics.pairwise import euclidean_distances as EuDist2
import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power
import random
def KNN(X,knn):
    eps = 2.2204e-16
    n, dim = X.shape
    D = EuDist2(X, X, squared=True)
    NN_full = np.argsort(D, axis=1)
    W = np.zeros((n,n))
    for i in range(n):
        id = NN_full[i,1:(knn+2)]
        di = D[i,id]
        W[i,id] = (di[-1]-di)/(knn*di[-1]-sum(di[:-1])+eps)
    A = (W+W.T)/2
    return A


def kng(X, knn, way="gaussian", t="mean", self=0, isSym=True):

    N, dim = X.shape
    # n x n graph
    D = EuDist2(X, X, squared=True)

    np.fill_diagonal(D, -1)
    NN_full = np.argsort(D, axis=1)
    np.fill_diagonal(D, 0)

    if self == 1:
        NN = NN_full[:, :knn]  # xi is among neighbors of xi
        NN_k = NN_full[:, knn]
    else:
        NN = NN_full[:, 1:(knn + 1)]  # xi isn't among neighbors of xi
        NN_k = NN_full[:, knn + 1]

    Val = get_similarity_by_dist(D=D, NN=NN, NN_k=NN_k, knn=knn, way=way, t=t)

    A = np.zeros((N, N))
    matrix_index_assign(A, NN, Val)

    if isSym:
        A = (A + A.T) / 2

    return A

def get_similarity_by_dist(D, NN, NN_k, knn, way, t):

    NND = matrix_index_take(D, NN)
    if way == "gaussian":
        if t == "mean":
            t = np.mean(D)
        elif t == "median":
            t = np.median(D)
        Val = np.exp(-NND / (2 * t ** 2))
    elif way == "t_free":
        NND_k = matrix_index_take(D, NN_k.reshape(-1, 1))
        Val = NND_k - NND
        ind0 = np.where(Val[:, 0] == 0)[0]
        if len(ind0) > 0:
            Val[ind0, :] = 1/knn
        Val = Val / (np.sum(Val, axis=1).reshape(-1, 1))
    else:
        raise SystemExit('no such options in "kng"')

    return Val

def matrix_index_take(X, ind_M):
    assert np.all(ind_M >= 0)
    n, k = ind_M.shape
    row = np.repeat(np.array(range(n), dtype=np.int32), k)
    col = ind_M.reshape(-1)
    ret = X[row, col].reshape((n, k))
    return ret


def matrix_index_assign(X, ind_M, Val):
    n, k = ind_M.shape
    row = np.repeat(np.array(range(n), dtype=np.int32), k)
    col = ind_M.reshape(-1)
    if isinstance(Val, (float, int)):
        X[row, col] = Val
    else:
        X[row, col] = Val.reshape(-1)

def norm_W(A):
    d = np.sum(A, 1)
    d[d == 0] = 1e-6
    d_inv = 1 / np.sqrt(d)
    tmp = A * np.outer(d_inv, d_inv)
    A2 = np.maximum(tmp, tmp.T)
    return A2
def ancher_W(X,knn,s):
    eps = 2.2204e-16
    N,_ = X.shape
    kmeans = KMeans(n_clusters=knn)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    # index = random.sample(range(N),knn)
    # centers = X[index]
    D = EuDist2(X,centers,squared=True)
    NN_full = np.argsort(D, axis=1)
    Z = np.zeros((N, knn))
    for i in range(N):
        id = NN_full[i, :s+1]
        di = D[i, id]
        Z[i, id] = (di[-1] - di) / (s * di[-1] - sum(di[:-1]) + eps)
    sum_zz = np.sum(Z,axis=0)
    lambda_z = np.diag(sum_zz)
    lambda_z_1 = fractional_matrix_power(lambda_z,-1)
    W = Z@lambda_z_1@Z.T
    return Z






