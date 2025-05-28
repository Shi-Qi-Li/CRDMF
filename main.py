import os
import time
import argparse
import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

from dataset import build_dataset
from model import build_model, reweight
from loss import build_loss, LossLog
from optim import build_optimizer
from metric import MetricLog, compute_metrics, compute_depth_criterion

from utils import write_scalar_to_tensorboard, set_random_seed, load_cfg_file, make_dirs, dict_to_log, init_logger


torch.autograd.set_detect_anomaly(True)

def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--config', required=True, help='the config file path')
    parser.add_argument('--mixed-precision', type=bool, default=False, help='use mixed precision training')

    args = parser.parse_args()
    return args

def train_step(dataset, model, optimizer, loss_func, iter, scaler, writer):
    model.train()
    train_loss = LossLog()
    
    optimizer.zero_grad()
    
    data = dataset()
    
    predictions = model()

    loss = loss_func(predictions, data)

    if scaler is None:        
        loss["loss"].backward()
        optimizer.step()
    else:
        scaler.scale(loss["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # results = summary_results("train", None, train_loss)
    if iter % 1000 == 0:
        write_scalar_to_tensorboard(writer, loss, iter)
    return loss

def val_step(dataset, model, val_step, writer):
    model.eval()
    val_metrics = MetricLog()
    with torch.no_grad():
        data = dataset()
    
        predictions = model()
        
        metrics = compute_metrics(predictions, data)
        criterion = compute_depth_criterion(predictions, data)

        metrics  = metrics | criterion

        val_metrics.add_metrics(metrics)

    # results = summary_results("val", val_metrics, None)
    write_scalar_to_tensorboard(writer, metrics, val_step)
    
    return metrics

def main():
    args = config_params()
    cfg = load_cfg_file(args.config)
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    experiment_dir = os.path.join(cfg.experiment_name, timestamp)
    make_dirs(experiment_dir)

    logger = init_logger(experiment_dir)
    dict_to_log(cfg, logger)

    set_random_seed(cfg.seed)

    loss_func = build_loss(cfg.loss)

    depth_candidate = load_cfg_file("config/depth_candidate.yaml")["depth_candidate"]

    model_list = []
    optimizer_list = []
    dataset_list = []
    scaler_list = []
    for depth in depth_candidate:
        
        cfg.model["depth"] = depth

        model = build_model(deepcopy(cfg.model))

        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = build_optimizer(model, deepcopy(cfg.optimizer))

        model_list.append(model)
        optimizer_list.append(optimizer)

        dataset = build_dataset(deepcopy(cfg.dataset)) if len(dataset_list) == 0 else deepcopy(dataset_list[-1]) 
        dataset_list.append(dataset)

        scaler = GradScaler() if args.mixed_precision else None
        scaler_list.append(scaler)
    
    
    if torch.cuda.is_available():
        loss_func = loss_func.cuda()

    scheduler = None

    summary_path = os.path.join("exp", experiment_dir, "summary")
    writer = SummaryWriter(summary_path)
    
    final_fit_mean = float("inf")
    final_result = None

    for iter in tqdm(range(cfg.iteration)):
        
        for model, optimizer, dataset, scaler, depth in zip(model_list, optimizer_list, dataset_list, scaler_list, depth_candidate):
            
            train_results = train_step(dataset, model, optimizer, loss_func, iter + 1, scaler, writer)     
            # print(train_results)
            if scheduler:
                scheduler.step()
            
            if (iter + 1) % cfg.interval == 0:
                val_results = val_step(dataset, model, (iter + 1) // cfg.interval, writer)
                
                logger.info("Depth {}, Iteration {}: {}".format(depth, iter + 1, val_results))
                
                fit_mean = val_results.get("Fit_mean", float("inf"))
                if (iter + 1) == cfg.iteration and fit_mean < final_fit_mean:
                    final_fit_mean = fit_mean
                    final_result = val_results

            if "reweight_interval" in cfg and (iter + 1) >= cfg.reweight_start and (iter + 1) % cfg.reweight_interval == 0:
                reweight(dataset, model)
    
    logger.info("Final result: {}".format(final_result))

    writer.close()

if __name__ == "__main__":
    main()
