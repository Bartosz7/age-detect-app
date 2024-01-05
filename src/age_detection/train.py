import os
import sys
from collections import defaultdict

import torch
import yaml
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from setup.data_manager import ImageData
from setup.model import resnet50
from setup.model_config import ModelConfig


def train(config_name: str, run_name: str, weights_path: str | None = None):
    config = ModelConfig.load_config(config_name, os.path.join(os.getcwd(), "configs"))

    checkpoint_folder = os.path.join(config.checkpoints_path, run_name)
    if os.path.exists(checkpoint_folder):
        raise ValueError(f"Checkpoint folder {checkpoint_folder} already exists.")
    
    os.makedirs(checkpoint_folder)
    model_folder = os.path.join(checkpoint_folder, "models")
    os.makedirs(model_folder)
    with open(
        os.path.join(checkpoint_folder, "config.yaml"), "w"
    ) as yaml_file:
        yaml.dump(config.__dict__, yaml_file, default_flow_style=False)
    
    train_dataset = ImageData(config, mode="train", auto_balancing=config.auto_balance)
    val_dataset = ImageData(config, mode="val")
    test_dataset = ImageData(config, mode="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=24,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=24,
    )
    #test_loader = torch.utils.data.DataLoader(
    #    test_dataset,
    #    batch_size=config.batch_size,
    #    shuffle=False,
    #    num_workers=24,
    #)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet50(weights_path, config.drop).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss = torch.nn.MSELoss().to(device)

    best_loss = float("inf")
    metrics = defaultdict(list)
    for epoch in range(config.epochs):
        train_loader.curr_epoch = epoch
        model.train()
        train_count = 0
        train_loss = 0
        for image, label in tqdm(train_loader, desc="train"):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(image)

            loss_value = loss(output, label)
            loss_value.backward()
            optimizer.step()
            train_count += 1
            train_loss += loss_value.item()
        
        metrics["train_loss"].append(train_loss / train_count)

        val_count = 0
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for image, label in tqdm(val_loader, desc="val"):
                image = image.to(device)
                label = label.to(device)

                output = model(image)

                loss_value = loss(output, label)
                val_count += 1
                val_loss += loss_value.item()

        metrics["val_loss"].append(val_loss / val_count)
        """
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
        
        metrics["test_loss"].append(test_loss / test_count)
        """
        print(
            f"Epoch [{epoch+1}/{config.epochs}], "
            f"Train Loss: {metrics['train_loss'][-1]:.4f}, "
            f"Val Loss: {metrics['val_loss'][-1]:.4f}, ",
            #f"Test Loss: {metrics['test_loss'][-1]:.4f}",
        )

        torch.save(
            model.state_dict(),
            os.path.join(model_folder, f"{epoch=}.pth"),
        )
        if metrics["val_loss"][-1] < best_loss:
            print(f"Best model saved: {best_loss:.4f} -> {metrics['val_loss'][-1]:.4f}")
            best_loss = metrics["val_loss"][-1]
            torch.save(model.state_dict(), os.path.join(checkpoint_folder, "best.pth"))
            
    print(f"Best model: {best_loss:.4f}")
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(checkpoint_folder, "data.csv"))

    plt.figure()
    plt.plot(metrics["train_loss"], label="train_loss")
    plt.plot(metrics["val_loss"], label="val_loss")
    #plt.plot(metrics["test_loss"], label="test_loss")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_folder, "losses.png"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <config_name> <run_name> <weights_path - optional>")
        sys.exit(1)

    weights_path = None
    if len(sys.argv) == 4:
        weights_path = sys.argv[3]
    train(sys.argv[1], sys.argv[2], weights_path)
