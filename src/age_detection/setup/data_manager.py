import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Literal
import pandas as pd
from .model_config import ModelConfig


class ImageData(Dataset):
    def __init__(self, config: ModelConfig, mode: Literal["train", "val", "test"]):
        self.images_path = config.images_path
        self.labels = pd.read_parquet(getattr(config, f"{mode}_labels_path"))
        self.mode = mode
        self.transform = {
            'train': transforms.Compose([                
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        }

    def __len__(self) -> int:
        """
        Get the length of the entire dataset
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the image item by index
        """
        image_path = os.path.join(self.images_path, self.labels.iloc[idx]["file_path"])
        image = Image.open(image_path)
        image_label = (self.labels.iloc[idx]["age"]-37) / 13
        transformed_img = self.transform[self.mode](image)

        return transformed_img, torch.Tensor([image_label])
    