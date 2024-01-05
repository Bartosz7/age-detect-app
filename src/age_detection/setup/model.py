import torch
import torch.nn as nn
import torchvision


def resnet50(weights_path: str | None = None, drop: float = 0.0):
    model = torchvision.models.resnet50(pretrained=weights_path is None)
    last_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(last_in_features, 128),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Linear(128, 1),
    )

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    
    return model
