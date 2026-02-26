"""
配置文件 - 包含所有超参数和训练设置
"""

import os
from datetime import datetime

class Config:
    # -------------------- 基础设置 --------------------
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
    
    EXP_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', EXP_NAME)
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # -------------------- 数据参数 --------------------
    # 使用CIFAR-10自动下载
    DATASET = 'cifar10'  # 'cifar10', 'cifar100', 'mnist', 'fashion_mnist', 或 'custom'
    
    # 如果是自定义数据集，设置以下路径
    TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'train')
    VAL_DATA_PATH = os.path.join(DATA_ROOT, 'val')
    TEST_DATA_PATH = os.path.join(DATA_ROOT, 'test')
    
    IMG_SIZE = 32 if DATASET == 'cifar10' else 224  # CIFAR-10是32x32
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # CIFAR-10有10个类
    NUM_CLASSES = 10
    
    # -------------------- 模型参数 --------------------
    MODEL_NAME = 'resnet18'
    PRETRAINED = False  # CIFAR-10太小，不需要预训练
    
    # -------------------- 训练参数 --------------------
    EPOCHS = 50
    DEVICE = 'cuda'
    
    OPTIMIZER = 'adam'
    BASE_LR = 0.001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    
    SCHEDULER = 'cosine'
    LR_DECAY_STEPS = [30, 40]
    LR_DECAY_GAMMA = 0.1
    
    WARMUP_EPOCHS = 5
    WARMUP_LR = 1e-6
    
    # -------------------- 损失函数 --------------------
    LOSS_TYPE = 'ce'
    
    # -------------------- 数据增强 --------------------
    USE_MOSAIC = False  # CIFAR-10图片太小，Mosaic效果不好
    MOSAIC_PROB = 0.5
    
    RANDOM_HORIZONTAL_FLIP_PROB = 0.5
    RANDOM_ROTATION_DEGREES = 15
    COLOR_JITTER = {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
    NORMALIZE_MEAN = [0.4914, 0.4822, 0.4465]  # CIFAR-10均值
    NORMALIZE_STD = [0.2470, 0.2435, 0.2616]   # CIFAR-10标准差
    
    # -------------------- 日志与保存 --------------------
    PRINT_FREQ = 50
    SAVE_FREQ = 5
    VAL_FREQ = 1
    
    @classmethod
    def make_dir(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.DATA_ROOT, exist_ok=True)
        
        config_path = os.path.join(cls.OUTPUT_DIR, 'config.txt')
        with open(config_path, 'w') as f:
            for key, value in cls.__dict__.items():
                if not key.startswith('__') and not callable(value):
                    f.write(f"{key}: {value}\n")
