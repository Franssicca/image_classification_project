"""
学习率调度模块
"""

import math
from torch.optim.lr_scheduler import _LRScheduler
from config.config import Config

class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing with warmup - 修复除零错误"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_lr=1e-6, base_lr=0.01, last_epoch=-1):
        self.warmup_epochs = max(warmup_epochs, 1)  # 确保不为0
        self.total_epochs = max(total_epochs, 1)    # 确保不为0
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * alpha
            return [lr for _ in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = max(0, min(1, progress))  # 限制在0-1之间
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            return [lr for _ in self.base_lrs]

def get_scheduler(optimizer):
    """获取学习率调度器"""
    if Config.SCHEDULER == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=Config.WARMUP_EPOCHS,
            total_epochs=Config.EPOCHS,
            warmup_lr=Config.WARMUP_LR,
            base_lr=Config.BASE_LR
        )
    else:
        scheduler = None
    
    return scheduler
