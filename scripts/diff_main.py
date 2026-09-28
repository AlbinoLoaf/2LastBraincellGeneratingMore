import argparse
import random
import numpy as np
import torch
import yaml

from dataset.datasets import LoadDataset
from diff_trainer import Trainer
from models.eegdiffuser import EEGDiffuser


class ConfigArgs:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="EEGDiffuser Main Pipeline")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to config file"
    )
    cmd_args = parser.parse_args()

    with open(cmd_args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    params = ConfigArgs(config_dict)
    setup_seed(params.seed)

    if torch.cuda.is_available():
        if params.cuda < torch.cuda.device_count():
            device = torch.device(f"cuda:{params.cuda}")
            torch.cuda.set_device(params.cuda)
        else:
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    load_dataset = LoadDataset(params)
    data_loader = load_dataset.get_data_loader()

    model = EEGDiffuser(num_classes=params.num_of_classes)
    trainer = Trainer(params=params, data_loader=data_loader, model=model)

    trainer.train()


if __name__ == "__main__":
    main()