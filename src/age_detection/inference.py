import os
import sys
import cv2
from functools import reduce

import torch
from PIL import Image
from torchvision import transforms

from .setup.model import resnet50
from .cut_out_face import get_face
from .setup.data_manager import MEAN_COLORS, STD_COLORS, STD_VALUE, MEAN_VALUE


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
        transforms.Normalize(mean=MEAN_COLORS, std=STD_COLORS),
    ])

    image = get_face(image_path)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)

    age = output.item() * STD_VALUE + MEAN_VALUE
    print(f"Predicted age: {age:.2f} years")


def predict_age_resnet50(model, device, image):

    if reduce(lambda x, y: x * y, image.shape) == 0:
        return 0

    image = Image.fromarray(image)  # Convert the NumPy array to PIL Image

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_COLORS, std=STD_COLORS),
    ])

    # image = get_face(image_path)
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = img_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    age = output.item() * STD_VALUE + MEAN_VALUE  # denormalization
    return int(age)


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
