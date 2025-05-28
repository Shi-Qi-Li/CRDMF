from typing import Dict, Optional

import os
import torch
import yaml
import logging
import numpy as np
from easydict import EasyDict as edict

from loss import LossLog
from metric import MetricLog

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def write_scalar_to_tensorboard(writer, results, iter):
    for key, value in results.items():
        if value != None:
            writer.add_scalar(key, value, iter)

def save_model(checkpoints_path, name, model_state):
    saved_path = os.path.join(checkpoints_path, "{}{}".format(name, ".pth")) 
    torch.save(model_state, saved_path)

def load_cfg_file(model_cfg_path: str):
    with open(model_cfg_path) as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    return edict(cfg)

def make_dirs(experiment_stamp: str, mode: str = "train"):
    work_dir = os.path.join("exp", experiment_stamp)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    summary_dir = os.path.join(work_dir, "summary")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if mode == "train":
        checkpoints_dir = os.path.join(work_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
def summary_results(mode: str, metrics: Optional[MetricLog] = None, loss: Optional[LossLog] = None):
    results = dict()

    if metrics is not None:
        for metric_category in metrics.all_metric_categories:
            results.update({''.join(["metrics_", mode, "/", metric_category]): metrics.get_metric(metric_category)})

    if loss is not None:
        for loss_category in loss.all_loss_categories:
            results.update({''.join(["loss_", mode, "/", loss_category]): loss.get_loss(loss_category)})

    return results

def to_cuda(data_batch: Dict):
    for key, value in data_batch.items():
        if isinstance(value, torch.Tensor):
            data_batch[key] = value.cuda()

def init_logger(experiment_dir: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(filename=os.path.join(os.path.join("exp", experiment_dir), "log.log"))
    streamhandler = logging.StreamHandler()
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    return logger

def dict_to_log(dictionary: Dict, logger: logging.Logger):
    for k, v in dictionary.items():
        logger.info("{}: {}".format(k, v))