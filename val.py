"""
验证脚本 - 评估模型在验证集上的性能
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data.dataset import CIFAR10Dataset
from data.transforms import ValTransforms
from models.model import get_model
from utils.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description='Model Validation')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    return parser.parse_args()

@torch.no_grad()
def validate(model, val_loader, device, logger):
    """验证模型"""
    model.eval()
    correct = 0
    total = 0
    
    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    logger.info(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    args = parse_args()
    
    logger = Logger('val')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = get_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 加载验证集
    val_dataset = CIFAR10Dataset(
        root=Config.DATA_ROOT,
        train=False,
        transform=ValTransforms(32)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    accuracy = validate(model, val_loader, device, logger)

if __name__ == '__main__':
    main()
