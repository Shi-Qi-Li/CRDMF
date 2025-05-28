import torch
import numpy as np
import graph_ops

from utils import convert_mat

def reweight(dataset, model):
    model.eval()
    
    with torch.no_grad():
        data = dataset()
    
        predictions = model()
        
        camera_num = data["camera_num"]

        if "pose" in predictions:
            pose = predictions["pose"].detach().clone().cpu().numpy()
            pose = pose.reshape(camera_num, 3, 3)

            U, _, V = np.linalg.svd(pose)
            pose = np.matmul(U, V)
            sign = np.linalg.det(pose)
            pose = pose * sign[:, None, None]
            R = pose.reshape(-1, 3)
        else:
            temp = predictions["e2e"].detach().clone().cpu().numpy()
            for i in range(0, temp.shape[0], 3):
                temp[i:i+3, i:i+3] = np.eye(3)
            R = convert_mat(temp, camera_num).transpose(2,0,1)
            R = R.reshape(-1, 3)

        w_fit = np.matmul(R, R.T).astype(np.float64)

        fit_error = graph_ops.fit_error(data["adjacent_mat"], camera_num, data["w_obs"], w_fit, "chordal")
        
        fit_error = fit_error.reshape(camera_num, camera_num).astype(np.float32)
        fit_error = np.max(np.stack([fit_error, fit_error.T]), axis=0)

        fit_error_median = np.median(fit_error[np.where(data["adjacent_mat"])])
        indices = np.where(fit_error > fit_error_median)
        small_weight_mat = np.ones((camera_num, camera_num))
        small_weight_mat[indices] = fit_error_median / fit_error[indices]
        fit_error = small_weight_mat

        new_weight = np.zeros((camera_num * 3, camera_num * 3))
        for i in range(3):
            for j in range(3):
                new_weight[i::3, j::3] = fit_error
        
        dataset.update_weight(new_weight)
    