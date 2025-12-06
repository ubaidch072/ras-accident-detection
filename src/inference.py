import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "best_model.pt"
CLASSES_FILE = PROJECT_ROOT / "classes.txt"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names():
    with open(CLASSES_FILE, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names


inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def load_trained_model():
    class_names = load_class_names()
    model = build_model(num_classes=len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names


def predict_image(image_path: Path):
    model, class_names = load_trained_model()

    img = Image.open(image_path).convert("RGB")
    tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]

    return pred_class, probs, class_names


def main():
    parser = argparse.ArgumentParser(description="Crash / Intact car classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    pred_class, probs, class_names = predict_image(image_path)
    print("Image:", image_path)
    print("Predicted class:", pred_class)
    print("Probabilities:")
    for cls, p in zip(class_names, probs):
        print(f"  {cls}: {p:.4f}")


if __name__ == "__main__":
    main()
