"""
模型定义模块
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config.config import Config

class ImageClassifier(nn.Module):
    """
    图像分类模型
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, model_name=Config.MODEL_NAME):
        super(ImageClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def get_model():
    return ImageClassifier()
