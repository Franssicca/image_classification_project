"""
优化器配置模块
"""

import torch.optim as optim
from config.config import Config

def get_optimizer(model):
    """获取优化器"""
    if Config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=Config.BASE_LR,
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY
        )
    elif Config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.BASE_LR,
            weight_decay=Config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unsupported optimizer: {Config.OPTIMIZER}")
    
    return optimizer
