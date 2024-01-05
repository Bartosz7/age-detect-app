import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Literal
from itertools import cycle
import pandas as pd
from .model_config import ModelConfig


MEAN_VALUE = 37.4
STD_VALUE = 14.5
MEAN_COLORS = [0.5646, 0.4326, 0.3711]
STD_COLORS = [0.2495, 0.2205, 0.2173]


class AddGaussianNoise:
    def __init__(self, mean:float = 0, std: float = 1, prob: float = 1):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, tensor):
        if torch.rand(1).item() < self.prob:
            return torch.clamp(tensor + torch.randn(tensor.size()) * self.std + self.mean, min=0, max=1)
        return tensor


class ImageData(Dataset):
    def __init__(self, config: ModelConfig, mode: Literal["train", "val", "test"], auto_balancing: bool = False):
        self.auto_balancing = auto_balancing
        self.images_path = config.images_path
        self.labels = pd.read_parquet(getattr(config, f"{mode}_labels_path"))
        self.mode = mode
        self.transform = {
            'train': transforms.Compose([                
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(0, 359), fill=0.5),
                transforms.Resize((224, 224)),
                transforms.ColorJitter(config.brightness, config.contrast, config.saturation, config.hue),
                transforms.ToTensor(),
                AddGaussianNoise(0, config.noise_std, config.noise_prob),
                transforms.Normalize(mean=MEAN_COLORS, std=STD_COLORS),
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_COLORS, std=STD_COLORS),
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_COLORS, std=STD_COLORS),
            ]),
        }

        self.curr_epoch = 0
        # Balancing labels
        self.label_counts = self.labels['age'].value_counts()
        self.num_classes = self.label_counts.shape[0]
        self.epoch_size_per_label = self.label_counts.min()
        self.mapping = []
        for _ in range(self.num_classes * self.epoch_size_per_label):
            self.mapping.append([])
        age_map = {age: [] for age in self.label_counts.index}

        assigned_values = 0
        for i, age in self.labels["age"].items():
            if len(age_map[age]) < self.epoch_size_per_label:
                age_map[age].append(assigned_values)
                self.mapping[assigned_values].append(i)
                assigned_values += 1
            else:
                v = age_map[age].pop()
                self.mapping[v].append(i)
                age_map[age].append(v)


    def __len__(self) -> int:
        # Length is now the total number of samples across all labels per epoch
        if self.auto_balancing:
            return self.num_classes * self.epoch_size_per_label
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the image item by index
        """
        if self.auto_balancing:
            balanced_idx = self.mapping[idx][self.curr_epoch % len(self.mapping[idx])]
        else:
            balanced_idx = idx
        image_path = os.path.join(self.images_path, self.labels.iloc[balanced_idx]["file_path"])
        image = Image.open(image_path)
        image_label = (self.labels.iloc[balanced_idx]["age"]-MEAN_VALUE) / STD_VALUE
        transformed_img = self.transform[self.mode](image)

        return transformed_img, torch.Tensor([image_label])
    