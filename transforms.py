"""
数据增强和预处理模块
"""

import random
import torchvision.transforms as T
from PIL import Image
from config.config import Config

class MosaicTransform:
    """
    Mosaic数据增强
    """
    def __init__(self, size=224, prob=0.5):
        self.size = size
        self.prob = prob
    
    def __call__(self, images, labels):
        if random.random() > self.prob:
            return images[0], labels[0]
        
        if len(images) < 4:
            return images[0], labels[0]
        
        s = self.size
        y_center = random.randint(s // 4, 3 * s // 4)
        x_center = random.randint(s // 4, 3 * s // 4)
        
        mosaic_img = Image.new('RGB', (s, s))
        
        positions = [
            (0, 0, x_center, y_center),
            (x_center, 0, s, y_center),
            (0, y_center, x_center, s),
            (x_center, y_center, s, s)
        ]
        
        for i, (img, (x1, y1, x2, y2)) in enumerate(zip(images, positions)):
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            
            img_resized = img.resize((w, h), Image.BILINEAR)
            mosaic_img.paste(img_resized, (x1, y1))
        
        mosaic_label = labels[random.randint(0, 3)]
        return mosaic_img, mosaic_label

class TrainTransforms:
    """训练集数据增强"""
    def __init__(self, size=32):
        self.size = size
        self.transforms = T.Compose([
            T.RandomCrop(size, padding=4),
            T.RandomHorizontalFlip(p=Config.RANDOM_HORIZONTAL_FLIP_PROB),
            T.ToTensor(),
            T.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])
    
    def __call__(self, img):
        return self.transforms(img)

class ValTransforms:
    """验证集/测试集数据增强"""
    def __init__(self, size=32):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])
    
    def __call__(self, img):
        return self.transforms(img)
