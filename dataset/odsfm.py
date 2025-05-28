import os
import numpy as np

import torch
import scipy.io
import graph_ops
from .builder import DATASET

@DATASET
class ODSFM:
    def __init__(self, data_path: str, scenario_name: str, init_method: str, delete_edge: bool):
        
        self.init_method = init_method

        obs_path = os.path.join(data_path, scenario_name + "_obs.pt")
        mat_path = os.path.join(data_path, scenario_name + ".mat")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        (self.us, self.vs), self.ys = torch.load(obs_path, map_location="cpu")
        self.us, self.vs, self.ys = self.us.numpy(), self.vs.numpy(), self.ys.numpy()
        self.ground_truth = scipy.io.loadmat(mat_path)['R_gt_c'].transpose(2,0,1)
        self.camera_num = scipy.io.loadmat(mat_path)['ncams_c'][0][0]

        obs_mat = np.zeros((self.camera_num * 3, self.camera_num * 3), dtype=np.int64)
        obs_mat[self.us, self.vs] = 1

        self.adjacent_mat = np.copy(obs_mat[::3,::3])
        assert np.all(self.adjacent_mat == self.adjacent_mat.T)
        print("[init]", np.sum(self.adjacent_mat))

        self.w_obs = np.eye(self.camera_num * 3, dtype=np.float64)

        self.w_obs[self.us, self.vs] = self.ys
        assert np.all(self.w_obs == self.w_obs.T)

        if delete_edge:
            support_tree = graph_ops.support_spanning_tree(np.copy(self.adjacent_mat), int(self.camera_num), np.copy(self.w_obs), "chordal")
            support_tree = support_tree.reshape(self.camera_num, self.camera_num)
            
            tree_view = graph_ops.get_pose_from_tree(self.camera_num, 0, support_tree, self.w_obs)
            tree_view = tree_view.reshape(self.camera_num, 3, 3)
            tree_view = tree_view.transpose(0, 2, 1)
            tree_view = tree_view.reshape(-1, 3)
            support_tree_obs = tree_view @ tree_view.T
            
            delta = graph_ops.fit_error(self.adjacent_mat, self.camera_num, self.w_obs, support_tree_obs, "chordal")
            delta = delta.reshape(self.camera_num, self.camera_num)
            self.adjacent_mat = np.logical_and(self.adjacent_mat, delta < 1).astype(np.int64)
            print("[deleted]", np.sum(self.adjacent_mat))

        if delete_edge:
            obs_mat = np.copy(self.adjacent_mat)
            obs_mat = np.expand_dims(obs_mat, axis=-1)
            obs_mat = np.repeat(obs_mat, repeats=9, axis=-1)
            obs_mat = obs_mat.reshape((self.camera_num, self.camera_num, 3, 3))
            obs_mat = obs_mat.transpose((0, 2, 1, 3))
            obs_mat = obs_mat.reshape((self.camera_num * 3, self.camera_num * 3))

            self.us, self.vs = np.where(obs_mat)
            
            self.ys = self.w_obs[self.us, self.vs]
            
        weight_mat = np.ones((self.camera_num * 3, self.camera_num * 3))

        weight = weight_mat[self.us, self.vs]
        weight *= weight.shape[0] / np.sum(weight)

        self.weight = torch.tensor(weight, device=self.device)
        self.us = torch.tensor(self.us, device=self.device)
        self.vs = torch.tensor(self.vs, device=self.device)
        self.ys = torch.tensor(self.ys, device=self.device, dtype=torch.float)
        self.ground_truth = torch.tensor(self.ground_truth, device=self.device)

    def update_weight(self, new_weight):
        new_weight = torch.tensor(new_weight, device=self.device)
        assert torch.all(torch.isnan(new_weight)) == False
        new_weight = new_weight[self.us, self.vs]

        assert self.weight.shape == new_weight.shape
        self.weight = self.weight * new_weight

    def __call__(self):
        
        data_batch = {
            "ys": self.ys,
            "us": self.us,
            "vs": self.vs,
            "ground_truth": self.ground_truth,
            "camera_num": self.camera_num,
            "weight": self.weight,
            "w_obs": self.w_obs,
            "adjacent_mat": self.adjacent_mat
        }

        return data_batch