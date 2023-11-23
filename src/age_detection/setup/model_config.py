from pydantic import BaseModel
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import os


class ModelConfig(BaseModel):
    images_path: str
    train_labels_path: str
    val_labels_path: str
    test_labels_path: str

    checkpoints_path: str

    epochs: int = 10
    batch_size: int = 32
    lr: float = 0.001

    @classmethod
    def load_config(cls, config_name: str, path: str) -> "ModelConfig":
        with initialize_config_dir(
            config_dir=path,
            job_name="setup config",
            version_base=None,
        ):
            cfg = compose(config_name=config_name)
            config = OmegaConf.to_object(cfg)
        return cls(**config)
