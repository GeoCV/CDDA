# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
from torch import nn

from torch.backends import cudnn
sys.path.append('.')
from config import cfg
from data import make_data_loader,make_data_loader2
from engine.trainer2 import do_train_with_center2
from modeling import build_model
from layers import make_loss, make_loss_with_center,make_loss_with_cluster
from solver import make_optimizer, make_optimizer_with_center,make_optimizer_with_center2, WarmupMultiStepLR

from utils.logger import setup_logger


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    target_train_loader, target_val_loader = make_data_loader2(cfg)

    # prepare model
    model = build_model(cfg, num_classes)

    
    print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
    loss_func, center_criterion = make_loss_with_center(cfg, num_classes)  # modified by gu
    cluster_num_classes = cfg.INPUT.CLUSTER_NUMBER
    loss_cluster_func,cluster_criterion = make_loss_with_cluster(cfg,cluster_num_classes)

    optimizer, optimizer_center, optimizer_cluster = make_optimizer_with_center2(cfg, model, center_criterion,cluster_criterion)
    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    arguments = {}
    
    """center_criterion = torch.load('D:\download\chromedownload\\resnet50_center_param_30 (1).pth')
    center_criterion.centers = nn.Parameter(center_criterion.centers)
    for param in center_criterion.parameters():
        param.grad.data *= (1. / center_loss_weight)"""

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
        print('Path to the checkpoint of center_param:', path_to_center_param)
        path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
        print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
        model = torch.load(cfg.MODEL.PRETRAIN_PATH)
        optimizer = torch.load(path_to_optimizer)
        center_criterion = torch.load(path_to_center_param)
        optimizer_center = torch.load(path_to_optimizer_center)
        ###
        if start_epoch >= cfg.SOLVER.MY_START_EPOCH:
            path_to_cluster_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'cluster_param')
            print('Path to the checkpoint of cluster_param:', path_to_cluster_param)
            path_to_optimizer_cluster = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_cluster')
            print('Path to the checkpoint of optimizer_cluster:', path_to_optimizer_cluster)
            cluster_criterion = torch.load(path_to_cluster_param)
            optimizer_cluster = torch.load(path_to_optimizer_cluster)
        ###
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                        cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                        cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))
    #trainr2.py
    do_train_with_center2(
        cfg,
        model,
        center_criterion,
        cluster_criterion,  #
        train_loader,
        val_loader,
        target_train_loader,#
        target_val_loader,#
        optimizer,
        optimizer_center,
        optimizer_cluster,  #
        scheduler,      # modify for using self trained model
        loss_func,
        loss_cluster_func,  #
        num_query,
        start_epoch,     # add for using self trained model
        cfg.SOLVER.MY_START_EPOCH #开始聚类损失的EPOCH
    )


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/new.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
