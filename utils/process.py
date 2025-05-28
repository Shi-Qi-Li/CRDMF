import numpy.typing as npt

import torch
import random
import scipy
import numpy as np

def convert_mat(mat: npt.NDArray, ncams: int):
    A = np.ones((ncams, ncams))
    D = np.kron(np.diag(1.0 / np.sum(A, 1)), np.eye(3))
    M = scipy.sparse.linalg.eigsh(np.matmul(D, mat), k=3)[1]
    M[:, [0, 2]] = M[:, [2, 0]]
    R = np.zeros((3, 3, ncams))

    for i in range(ncams):
        U, _, V = np.linalg.svd(M[3*i:3*i + 3, :])
        R[:, :, i] = np.matmul(U, V)
        if (np.linalg.det(R[:, :, i]) < 0):
            R[:, :, i] = - np.matmul(U, V)

    return R

def R2w(R):
    w = torch.stack((R[2, 1] - R[1, 2], R[0, 2] -
                    R[2, 0], R[1, 0] - R[0, 1])) / 2
    s = torch.norm(w)
    if s:
        w = w / s * torch.atan2(s, (torch.trace(R) - 1) / 2)
    return w

def w2R(w):
    omega = torch.norm(w)
    if omega:
        n = w / omega
        s = torch.sin(omega)
        c = torch.cos(omega)
        cc = 1 - c
        n1 = n[0]
        n2 = n[1]
        n3 = n[2]
        n12cc = n1*n2*cc
        n23cc = n2*n3*cc
        n31cc = n3*n1*cc
        n1s = n1*s
        n2s = n2*s
        n3s = n3*s
        R = torch.zeros(3, 3)
        R[0, 0] = c+n1*n1*cc
        R[0, 1] = n12cc-n3s
        R[0, 2] = n31cc+n2s
        R[1, 0] = n12cc+n3s
        R[1, 1] = c+n2*n2*cc
        R[1, 2] = n23cc-n1s
        R[2, 0] = n31cc-n2s
        R[2, 1] = n23cc+n1s
        R[2, 2] = c+n3*n3*cc
    else:
        R = torch.eye(3)
    R = R.to(torch.float64)
    R = R.to(w.device)
    return R

def pose_grpah_pairwise_error(w_gt: npt.NDArray, w_obs: npt.NDArray) -> npt.NDArray:

    assert w_gt.shape == w_obs.shape

    camera_num = w_gt.shape[0] // 3

    w_obs = w_obs.reshape(camera_num, 3, camera_num, 3)
    w_obs = w_obs.swapaxes(1, 2)

    w_gt = w_gt.reshape(camera_num, 3, camera_num, 3)
    w_gt = w_gt.swapaxes(1, 2)

    error = np.trace(np.matmul(w_obs, w_gt.swapaxes(-1, -2)), axis1=-2, axis2=-1)
    error = (error - 1) / 2
    error = np.clip(error, a_min=-1, a_max=1)

    error = np.arccos(error)
    error = np.rad2deg(error)

    return error

def faster_compare_rot_graph(R1, R2, method="median"):
    torchpi = torch.acos(torch.zeros(1)).item() * 2
    sigma2 = (5 * torchpi / 180) * (5 * torchpi / 180)
    N = R1.shape[0]
    Emeanbest = float("Inf")
    E = torch.zeros(3).to(R1.device)
    Ebest = E.clone()
    e = torch.zeros(N, 1).to(R1.device)
    ebest = e
    for i in range(4):
        j = random.randint(0, N-1)
        
        R = R1[j, :, :].clone().t()
        R1 = torch.matmul(R1, R)
        R = R2[j, :, :].clone().t()
        R2 = torch.matmul(R2, R)
        W = torch.zeros(N, 3).to(R1.device)
        d = float("Inf")
        count = 1
        while(d > 1e-5 and count < 20):
            W = batch_R2w(torch.matmul(R2.transpose(-1, -2), R1))
            if method == "mean":
                w = torch.mean(W, 0)
                d = torch.norm(w)
                R = w2R(w)
            elif method == "median":
                w = torch.median(W, 0).values
                d = torch.norm(w)
                R = w2R(w)
            elif method == "robustmean":
                w = 1 / torch.sqrt(torch.sum(W * W, 1) + sigma2)
                w = w/torch.sum(w)
                w = torch.mean(w.repeat(1, 3) * W)
                d = torch.norm(w)
                R = w2R(w)
            R2 = torch.matmul(R2, R)
            count = count + 1
        
        now = (torch.vmap(torch.trace)(torch.matmul(R1, R2.transpose(-1, -2))) - 1) / 2
        now = now.unsqueeze(dim=-1)
        now = torch.clamp(now, -1, 1)
        e = torch.acos(now)
        e = torch.rad2deg(e)
        
        E = torch.stack([torch.mean(e), torch.median(
            e), torch.sqrt(torch.mm(e.t(), e)/len(e))[0, 0]])
        if E[0] < Emeanbest:
            Ebest = E
            Emeanbest = E[0]

    E_mean, E_median, E_var = Ebest[0].item(), Ebest[1].item(), Ebest[2].item()
    
    return E_mean, E_median, E_var

def batch_R2w(R):
    w = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] -
                    R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], dim=-1) / 2
    s = torch.norm(w, dim=-1)

    index = torch.nonzero(s > 0)
    index = index.reshape(-1)

    if index.shape[0] > 0:

        nonzero_w = torch.index_select(w, dim=0, index=index)
        nonzero_s = torch.index_select(s, dim=0, index=index)
        nonzero_R = torch.index_select(R, dim=0, index=index)

        nonzero_w = nonzero_w / nonzero_s.unsqueeze(dim=-1) * torch.atan2(nonzero_s, (torch.vmap(torch.trace)(nonzero_R) - 1) / 2).unsqueeze(dim=-1)

        w[index] = nonzero_w

    return w