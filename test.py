import argparse
import json
import os
import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from timm.models import create_model

import model.build

# Default ImageNet normalization constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# URL for standard ImageNet class labels
IMAGENET_CLASSES_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)
DEFAULT_LABEL_FILE = "imagenet_classes.txt"


def download_labels(url=IMAGENET_CLASSES_URL, outfile=DEFAULT_LABEL_FILE):
    """Downloads the ImageNet class labels file if it doesn't exist."""
    if not os.path.exists(outfile):
        print(f"Downloading ImageNet labels from {url} to {outfile}...")
        urlretrieve(url, outfile)
        print("Download complete.")
    else:
        print(f"Label file {outfile} already exists.")


def load_labels(label_file=DEFAULT_LABEL_FILE):
    """Loads labels from a text file."""
    if not os.path.exists(label_file):
        raise FileNotFoundError(
            f"Label file not found: {label_file}. Please download it or provide the correct path."
        )
    with open(label_file) as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def get_preprocessing_transforms(input_size=224):
    """Creates standard ImageNet validation transforms."""
    # Note: Typically uses Resize(256) then CenterCrop(224),
    # but for direct inference, resizing to input_size might be sufficient.
    # Let's use Resize and CenterCrop similar to validation pipelines.
    scale_size = int(input_size / 0.875)  # Standard practice: resize slightly larger
    return transforms.Compose(
        [
            transforms.Resize(
                scale_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def predict(args):
    """Loads model, preprocesses image, predicts label, and displays."""

    # ---- 1. Setup Device ----
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- 2. Load Labels ----
    if args.download_labels:
        download_labels(outfile=args.label_file)
    try:
        labels = load_labels(label_file=args.label_file)
        num_classes = len(labels)
        print(f"Loaded {num_classes} labels.")
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    # ---- 3. Load Model ----
    print(f"Creating model: {args.model}")
    # Create model without distillation head if it exists, specify num classes
    model = create_model(
        args.model,
        num_classes=num_classes,
        pretrained=False,  # Load weights manually
        distillation=False,  # Assume no distillation head for inference
        fuse=False,  # Set based on original code defaults if needed
    )

    # ---- 4. Load Checkpoint ----
    if not args.checkpoint:
        print("Error: Checkpoint path must be provided (--checkpoint)")
        return
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return

    print(f"Loading checkpoint from: {args.checkpoint}")
    # Load checkpoint to CPU first to avoid GPU memory issues with metadata
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Extract the state dict - check common keys
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # Assume the checkpoint file is just the state dict

    # Handle potential DataParallel/DistributedDataParallel 'module.' prefix
    if all(key.startswith("module.") for key in state_dict):
        print("Removing 'module.' prefix from checkpoint keys...")
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    # Load the state dict into the model
    msg = model.load_state_dict(
        state_dict, strict=False
    )  # Use strict=False if heads might differ
    print(f"Checkpoint loading message: {msg}")

    model.to(device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded and set to evaluation mode.")

    # ---- 5. Load and Preprocess Image ----
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return

    try:
        img_pil = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image file: {e}")
        return

    preprocess = get_preprocessing_transforms(input_size=args.input_size)
    input_tensor = preprocess(img_pil)
    input_batch = input_tensor.unsqueeze(
        0
    )  # Create a mini-batch as expected by the model
    input_batch = input_batch.to(device)
    print("Image loaded and preprocessed.")

    # ---- 6. Perform Inference ----
    print("Performing inference...")
    with torch.no_grad():
        output = model(input_batch)

    # ---- 7. Post-process Output ----
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_idx = torch.topk(probabilities, 1)

    predicted_idx = top1_idx[0].item()
    predicted_label = labels[predicted_idx]
    predicted_prob = top1_prob[0].item()

    print(f"\nPrediction:")
    print(f"  Label: {predicted_label} (Index: {predicted_idx})")
    print(f"  Confidence: {predicted_prob:.4f}")

    # ---- 8. Display Image with Label using OpenCV ----
    try:
        img_cv = cv2.imread(args.image_path)
        if img_cv is None:
            raise IOError("Could not read image with OpenCV")

        # Prepare text and position
        text = f"{predicted_label} ({predicted_prob:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (0, 255, 0)  # Green
        bg_color = (0, 0, 0)  # Black background for text

        # Get text size to draw background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        text_x = 10
        text_y = 30  # Position from top-left

        # Draw background rectangle
        cv2.rectangle(
            img_cv,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            bg_color,
            -1,
        )  # -1 fills the rectangle

        # Put text on top of rectangle
        cv2.putText(
            img_cv, text, (text_x, text_y), font, font_scale, text_color, font_thickness
        )

        # Display the image
        cv2.imshow("SHViT Prediction", img_cv)
        print("\nDisplaying image with prediction. Press any key to close.")
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()  # Close the window

    except Exception as e:
        print(f"\nError displaying image with OpenCV: {e}")
        print("Prediction results are still available above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHViT Image Prediction Script")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the SHViT model (e.g., shvit_s4)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--image-path", type=str, required=True, help="Path to the input image file"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size (e.g., 224 or 256)",
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default=DEFAULT_LABEL_FILE,
        help="Path to the ImageNet labels text file",
    )
    parser.add_argument(
        "--download-labels",
        action="store_true",
        help=f"Download ImageNet labels to {DEFAULT_LABEL_FILE} if not found",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cuda' or 'cpu')",
    )

    args = parser.parse_args()
    predict(args)
