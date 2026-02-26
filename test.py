"""
测试/推理脚本
"""

import os
import sys
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from models.model import get_model
from data.transforms import ValTransforms

def parse_args():
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image-path', type=str, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    return parser.parse_args()

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    original = image.copy()
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor, original

def predict(model, image_tensor, device, top_k=5):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
    return top_probs.cpu().numpy()[0], top_indices.cpu().numpy()[0]

def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_model(args.checkpoint, device)
    transform = ValTransforms(32)
    
    input_tensor, original = preprocess_image(args.image_path, transform)
    top_probs, top_indices = predict(model, input_tensor, device, args.top_k)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nPredictions:")
    for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        print(f"  {j+1}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")

if __name__ == '__main__':
    main()
