"""
日志记录模块
"""

import os
import logging
from datetime import datetime
from config.config import Config

class Logger:
    """日志记录器"""
    def __init__(self, name='train'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        log_file = os.path.join(Config.LOG_DIR, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):  # 👈 新增这个方法
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)

class AverageMeter:
    """计算平均值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
