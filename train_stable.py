#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定训练脚本 - 解决梯度爆炸问题
"""

import os
import sys
import torch
import logging
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import TrainDataset, TestDataset, ValiDataset
from torch.utils.data.sampler import RandomSampler

import utils
from kernel import train_and_evaluate, evaluate
import warnings
warnings.filterwarnings('ignore')

# 动态导入模型
def get_model_module(model_type):
    if model_type == 'QAspline':
        from model import net_qspline_A as net
    elif model_type == 'QBspline':
        from model import net_qspline_B as net
    elif model_type == 'QABspline':
        from model import net_qspline_AB as net
    elif model_type == 'QCDspline':
        from model import net_qspline_C as net
    elif model_type == 'Lspline':
        from model import net_lspline as net
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return net

def run_stable_training(model_dir='base_model'):
    """运行稳定训练"""
    
    # 加载配置
    params = utils.Params(os.path.join(model_dir, 'params.json'))
    dirs = utils.Params(os.path.join(model_dir, 'dirs.json'))
    
    # 设置随机种子
    utils.seed(42)
    
    # 设置日志
    utils.set_logger(os.path.join(model_dir, 'train_stable.log'))
    logger = logging.getLogger('DeepAR.Train')
    
    # 更新目录路径
    dirs = utils.dirs_update(dirs)
    
    # 检查CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dirs.device = torch.device('cuda:0')
        logger.info('Using Cuda...')
    else:
        dirs.device = torch.device('cpu')
        logger.info('Not using cuda...')
    
    # 获取模型模块
    net = get_model_module(params.line)
    
    # 创建模型
    if use_cuda:
        model = net.Net(params, dirs.device).cuda(dirs.device)
    else:
        model = net.Net(params, dirs.device)
    
    logger.info(f'Model: \n{str(model)}')
    
    # 加载数据
    logger.info('Loading datasets...')
    train_set = TrainDataset(dirs.data_dir, dirs.dataset)
    vali_set = ValiDataset(dirs.data_dir, dirs.dataset)
    test_set = TestDataset(dirs.data_dir, dirs.dataset)
    
    train_loader = DataLoader(train_set, batch_size=params.batch_size, pin_memory=False, num_workers=4)
    vali_loader = DataLoader(vali_set, batch_size=params.batch_size, pin_memory=False, 
                            sampler=RandomSampler(vali_set), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.batch_size, pin_memory=False,
                            sampler=RandomSampler(test_set), num_workers=4)
    
    logger.info('Data loading complete.')
    
    # 设置优化器 - 使用更稳定的配置
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=1e-5)
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=10, verbose=True)
    
    # 训练
    logger.info(f'Starting stable training for {params.num_epochs} epochs')
    logger.info(f'Learning rate: {params.lr}')
    logger.info(f'Gradient clipping norm: {getattr(params, "grad_clip_norm", 1.0)}')
    
    best_test_CRPS = train_and_evaluate(model, train_loader, vali_loader, 
                                       optimizer, net.loss_fn, params, dirs, None)
    
    # 加载最佳模型进行测试
    load_dir = os.path.join(dirs.model_save_dir, 'best.pth.tar')
    if os.path.exists(load_dir):
        utils.load_checkpoint(load_dir, model)
        test_results = evaluate(model, net.loss_fn, test_loader, params, dirs, istest=True)
        test_json_path = os.path.join(model_dir, 'test_results_stable.json')
        utils.save_dict_to_json(test_results, test_json_path)
        logger.info(f'Final test CRPS: {test_results["CRPS_Mean"]:.6f}')
    
    logger.handlers.clear()
    logging.shutdown()
    
    return best_test_CRPS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='base_model', help='Model directory')
    args = parser.parse_args()
    
    run_stable_training(args.model_dir) 