"""
数据集加载模块 - 支持自动下载CIFAR-10
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from config.config import Config
from data.transforms import TrainTransforms, ValTransforms

class CIFAR10Dataset(Dataset):
    """
    支持增强的CIFAR-10数据集（自动下载）
    """
    def __init__(self, root='./data', train=True, transform=None):
        self.transform = transform
        self.train = train
        
        # 自动下载CIFAR-10！
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,  # 关键：自动下载
            transform=None
        )
        
        # 按类别组织索引，方便Mosaic采样
        self.indices_by_class = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in self.indices_by_class:
                self.indices_by_class[label] = []
            self.indices_by_class[label].append(idx)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
    def get_mosaic_item(self, idx):
        """
        获取用于Mosaic增强的4张图像
        """
        # 获取主图像
        main_img, main_target = self.dataset[idx]
        
        images = [main_img]
        labels = [main_target]
        
        # 随机选择另外3张图
        for _ in range(3):
            # 50%概率选择同类
            if random.random() < 0.5 and len(self.indices_by_class[main_target]) > 1:
                other_idx = random.choice(self.indices_by_class[main_target])
                while other_idx == idx and len(self.indices_by_class[main_target]) > 1:
                    other_idx = random.choice(self.indices_by_class[main_target])
            else:
                other_idx = random.randint(0, len(self.dataset) - 1)
            
            other_img, other_target = self.dataset[other_idx]
            images.append(other_img)
            labels.append(other_target)
        
        return images, labels

def get_dataloaders():
    """
    创建训练、验证数据加载器
    """
    # 训练集 - 自动下载
    train_dataset = CIFAR10Dataset(
        root=Config.DATA_ROOT,
        train=True,
        transform=TrainTransforms(Config.IMG_SIZE)
    )
    
    # 验证集 - 自动下载
    val_dataset = CIFAR10Dataset(
        root=Config.DATA_ROOT,
        train=False,
        transform=ValTransforms(Config.IMG_SIZE)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # CIFAR-10类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, val_loader, class_names
