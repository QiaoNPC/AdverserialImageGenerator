import argparse
import sys
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torch import nn
from torchvision import transforms
import os
import fgsm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_helpful_message(parser):
    parser.print_help()
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FGSM Brute Force Attack")
    parser.add_argument("--model", type=str, help="Hugging Face model name", required=True)
    parser.add_argument("--image", type=str, help="Path to the input image", required=True)
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon value for the attack (default: 0.05)")
    parser.add_argument("--size", type=int, help="Size of the input image for the model", required=True)
    parser.add_argument("--target", type=str, help="Target class name", required=True)
    parser.add_argument("--d", type=str, help="Directory to save adversarial images", required=True)

    if len(sys.argv) == 1:
        print_helpful_message(parser)

    args = parser.parse_args()

    fgsm.fgsm_brute_force(
        hugging_face_model=args.model,
        image_path=args.image,
        epsilon=args.epsilon,
        model_image_size=args.size,
        target_class_name=args.target,
        directory=args.d
    )
