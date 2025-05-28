from typing import Dict

import torch
import numpy as np

from utils import convert_mat, pose_grpah_pairwise_error, faster_compare_rot_graph

def compute_metrics(predictions: Dict, ground_truth: Dict) -> Dict:
    metrics = dict()
    
    method = "median"
    e2e = predictions["e2e"]
    gt = ground_truth["ground_truth"].detach().clone()
    camera_num = ground_truth["camera_num"]
    adjacent_mat = ground_truth["adjacent_mat"]

    temp = e2e.detach().clone().cpu().numpy()
    prerank = np.linalg.matrix_rank(temp)
    for i in range(0, temp.shape[0], 3):
        temp[i:i+3 , i:i+3] = np.eye(3)
    rank = np.linalg.matrix_rank(temp)
    
    if "pose" in predictions:
        pose = predictions["pose"].detach().clone().cpu().numpy()
        pose = pose.reshape(camera_num, 3, 3)

        U, _, V = np.linalg.svd(pose)
        pose = np.matmul(U, V)
        sign = np.linalg.det(pose)
        pose = pose * sign[:, None, None]

        pose = torch.from_numpy(pose).to(e2e.device).to(torch.float64)
        
        E_mean, E_median, E_var = faster_compare_rot_graph(pose, gt, method = method)
    else:
        R = torch.from_numpy(convert_mat(temp, camera_num).transpose(2,0,1)).to(e2e.device)
        E_mean, E_median, E_var = faster_compare_rot_graph(R, gt, method = method)
    
    temp = temp.reshape(camera_num, 3, camera_num, 3)
    temp = temp.transpose(0, 2, 1, 3)
    
    U, _, V = np.linalg.svd(temp)
    temp = np.matmul(U, V)
    covert = np.where(np.linalg.det(temp) < 0, -np.ones_like(adjacent_mat), np.ones_like(adjacent_mat))

    temp = temp * covert.reshape(camera_num, camera_num, 1, 1)
    temp = temp.transpose(0, 2, 1, 3)
    temp = temp.reshape(camera_num * 3, camera_num * 3)
    
    metrics["E_mean"] = E_mean
    metrics["E_median"] = E_median
    metrics["E_var"] = E_var
    metrics["rank"] = rank
    metrics["prerank"] = prerank

    return metrics

def compute_depth_criterion(predictions: Dict, ground_truth: Dict) -> Dict:
    metrics = dict()
    
    e2e = predictions["e2e"]
    camera_num = ground_truth["camera_num"]
    w_obs = ground_truth["w_obs"]
    adjacent_mat = ground_truth["adjacent_mat"]

    temp = e2e.detach().clone().cpu().numpy()
    for i in range(0, temp.shape[0], 3):
        temp[i:i+3 , i:i+3] = np.eye(3)
    rank = np.linalg.matrix_rank(temp)
    
    temp = temp.reshape(camera_num, 3, camera_num, 3)
    temp = temp.transpose(0, 2, 1, 3)
    
    U, _, V = np.linalg.svd(temp)
    temp = np.matmul(U, V)
    covert = np.where(np.linalg.det(temp) < 0, -np.ones_like(adjacent_mat), np.ones_like(adjacent_mat))

    temp = temp * covert.reshape(camera_num, camera_num, 1, 1)
    temp = temp.transpose(0, 2, 1, 3)
    temp = temp.reshape(camera_num * 3, camera_num * 3)

    pairwise_fit_error = pose_grpah_pairwise_error(w_obs, temp)
    fit_error = pairwise_fit_error[np.where(adjacent_mat - np.eye(camera_num))]
    
    metrics["Fit_mean"] = np.mean(fit_error)
    metrics["Fit_median"] = np.median(fit_error)
    
    metrics["rank"] = rank

    return metrics