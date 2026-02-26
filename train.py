"""
主训练脚本
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data.dataset import get_dataloaders
from models.model import get_model
from utils.loss import get_loss_function
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.logger import Logger, AverageMeter

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification Training')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()

def update_config_from_args(args):
    if args.epochs:
        Config.EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.BASE_LR = args.lr
    if args.device:
        Config.DEVICE = args.device

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, logger):
    model.train()
    losses = AverageMeter()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(Config.DEVICE)
        targets = targets.to(Config.DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        
        if (batch_idx + 1) % Config.PRINT_FREQ == 0:
            logger.info(
                f"Epoch: {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {losses.avg:.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
    
    return losses.avg

def validate(model, val_loader, criterion, logger):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            losses.update(loss.item(), images.size(0))
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return losses.avg, accuracy

def save_checkpoint(model, optimizer, epoch, accuracy, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(Config.CHECKPOINT_DIR, 'best.pth')
        torch.save(checkpoint, best_path)

def main():
    args = parse_args()
    update_config_from_args(args)
    Config.make_dir()
    
    logger = Logger('train')
    logger.info("=" * 50)
    logger.info("Starting Image Classification Training")
    logger.info("=" * 50)
    
    if Config.DEVICE == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        Config.DEVICE = 'cpu'
    logger.info(f"Using device: {Config.DEVICE}")
    
    logger.info("Loading CIFAR-10 dataset (auto-downloading)...")
    train_loader, val_loader, class_names = get_dataloaders()
    logger.info(f"Number of classes: {len(class_names)}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    logger.info(f"Creating model: {Config.MODEL_NAME}")
    model = get_model()
    model = model.to(Config.DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    criterion = get_loss_function()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    
    best_acc = 0.0
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, logger)
        
        if scheduler:
            scheduler.step()
        
        if epoch % Config.VAL_FREQ == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, logger)
            logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                logger.info(f"New best accuracy: {best_acc:.2f}%")
            
            save_checkpoint(model, optimizer, epoch, val_acc, is_best)
    
    logger.info("=" * 50)
    logger.info(f"Training completed!")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")
    logger.info("=" * 50)

if __name__ == '__main__':
    main()
