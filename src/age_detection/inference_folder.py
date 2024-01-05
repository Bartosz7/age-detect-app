import os
import sys
import cv2
import glob
import numpy as np

import torch
from PIL import Image
from torchvision import transforms

from setup.model import resnet50
from cut_out_face import get_face
from setup.data_manager import MEAN_COLORS, STD_COLORS, STD_VALUE, MEAN_VALUE


def inference(run_path: str, image_folder: str, model_id: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(run_path, "best.pth")
    if model_id is not None:
        model_path = os.path.join(run_path, "models", f"epoch={model_id}.pth")
    model = resnet50(model_path).to(device)
    model.eval()

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_COLORS, std=STD_COLORS),
    ])

    predictions = []
    for image_path in glob.glob(os.path.join(image_folder, "*")):
        image = get_face(image_path)
        if image is None:
            print(f"No image - {os.path.split(image_path)[1]}")
            continue
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = img_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)

        age = output.item() * STD_VALUE + MEAN_VALUE
        predictions.append(age)
        print(f"Predicted age: {age:.2f} years - {os.path.split(image_path)[1]}")

    predictions = np.array(predictions)
    print("Avg: ", predictions.mean(), "Std: ", predictions.std())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <run_path> <image_folder> <model_id - optional>")
        exit(1)
    run_path = sys.argv[1]
    image_folder = sys.argv[2]
    model_id = None
    if len(sys.argv) == 4:
        model_id = int(sys.argv[3])
    inference(run_path, image_folder, model_id)
