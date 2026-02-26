"""
损失函数模块
"""

import torch.nn as nn
from config.config import Config

def get_loss_function():
    """获取损失函数"""
    return nn.CrossEntropyLoss()
