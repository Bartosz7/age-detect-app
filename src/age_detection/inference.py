import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from setup.model import resnet50


def inference(run_path: str, image_path: str, model_id: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(run_path, "best.pth")
    if model_id is not None:
        model_path = os.path.join(run_path, "models", f"epoch={model_id}.pth")
    model = resnet50(model_path).to(device)
    model.eval()

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    image = img_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    age = output.item() * 13 + 37
    print(f"Predicted age: {age:.2f} years")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <run_path> <image_path> <model_id - optional>")
        exit(1)
    run_path = sys.argv[1]
    image_path = sys.argv[2]
    model_id = None
    if len(sys.argv) == 4:
        model_id = int(sys.argv[3])
    inference(run_path, image_path, model_id)
