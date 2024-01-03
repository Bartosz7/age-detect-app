import os
import sys

import torch
from tqdm import tqdm

from setup.model_config import ModelConfig
from setup.data_manager import ImageData
from setup.model import resnet50


def test(run_path: str, model_id: int | None = None):
    config = ModelConfig.load_config("config.yaml", run_path)

    test_dataset = ImageData(config, mode="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(run_path, "best.pth")
    if model_id is not None:
        model_path = os.path.join(run_path, "models", f"epoch={model_id}.pth")
    model = resnet50(model_path).to(device)
    model.eval()

    loss = torch.nn.L1Loss().to(device)

    test_count = 0
    test_loss = 0
    with torch.no_grad():
        for image, label in tqdm(test_loader, desc="test"):
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            loss_value = loss(output, label)
            test_count += 1
            test_loss += loss_value.item()

    print(f"Test accuracy: {test_loss / test_count * 13:.2f} years")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <run_path> <model_id - optional>")
        exit(1)

    run_path = sys.argv[1]
    model_id = None
    if len(sys.argv) == 3:
        model_id = int(sys.argv[2])
    test(run_path, model_id)
