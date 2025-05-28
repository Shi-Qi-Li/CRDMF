from typing import Dict

import torch
import torch.nn as nn
import numpy as np

from .builder import MODEL


@MODEL
class Half_MLP(nn.Module):
    def __init__(self, dim: int, depth: int, init_scale: float, use_svd: bool = False) -> None:
        super().__init__()

        self.dim = dim
        self.use_svd = use_svd
        
        self.mlp = nn.Sequential()
        for i in range(depth):
            if i == depth - 1:
                self.mlp.add_module("layer_{}".format(i + 1), nn.Linear(dim, 3, bias=False))
            else:
                self.mlp.add_module("layer_{}".format(i + 1), nn.Linear(dim, dim, bias=False))

        scale = init_scale**(1. / (depth * 2)) * dim**(-0.5)
        for param in self.parameters():
            if param.shape[0] == param.shape[1]:
                nn.init.normal_(param, std=scale)
            else:
                nn.init.normal_(param, std=scale * (dim / 3)**(0.25))
                
        e2e = self.get_e2e(self.get_pose()).detach().cpu().numpy()
        e2e_fro = np.linalg.norm(e2e, 'fro')
        desired_fro = init_scale * np.sqrt(dim)
        print(f"[check] e2e fro norm: {e2e_fro:.6e}, desired = {desired_fro:.6e}")
        
    def forward(self) -> torch.Tensor:        
        
        pose = self.get_pose(self.use_svd)
        e2e = self.get_e2e(pose)

        predictions = {
            "e2e": e2e,
            "pose": pose
        }
        
        return predictions

    def create_ground_truth(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "ys": data["ys"],
            "us": data["us"],
            "vs": data["vs"],
            "ground_truth": data["ground_truth"],
            "camera_num": data["camera_num"]
        }
    
    def get_e2e(self, pose):
        e2e = pose @ pose.t()

        return e2e

    def get_pose(self, use_svd: bool = False):
        pose = None
        for fc in self.mlp.children():
            assert isinstance(fc, nn.Linear) and fc.bias is None
            if pose is None:
                pose = fc.weight.t()
            else:
                pose = fc(pose)

        if pose.shape[-1] != 3:
            pose = pose.transpose(-1, -2)

        if use_svd:
            pose = pose.reshape(self.dim // 3, 3, 3)

            U, _, V = torch.svd(pose)
            R = torch.matmul(U, V)
            det = torch.linalg.det(R)
            sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
            R = R * sign.reshape(-1, 1, 1)

            pose = pose.reshape(-1, 3)
        
        return pose