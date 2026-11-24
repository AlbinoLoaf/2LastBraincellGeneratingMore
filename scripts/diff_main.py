import argparse
import random
import numpy as np
import torch
import yaml

from dataset.datasets import LoadDataset
from diff_trainer import Trainer
from diffusion import create_diffusion
from models.eegdiffuser import EEGDiffuser as CodeEEGDiffuser
from models.paper_eegdiffuser import EEGDiffuser as PaperEEGDiffuser


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
    parser = argparse.ArgumentParser(description="EEGDiffuser Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    cmd_args = parser.parse_args()

    with open(cmd_args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    if "real_data_dir" in config_dict and "datasets_dir" not in config_dict:
        config_dict["datasets_dir"] = config_dict["real_data_dir"]

    params = ConfigArgs(config_dict)
    setup_seed(params.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    arch_mode = getattr(params, "arch_mode", "paper")
    print(f"--- DIFFUSION TRAINING | ARCH: {arch_mode.upper()} | MODE: {params.diffusion_mode.upper()} | DEVICE: {device} ---")

    load_dataset = LoadDataset(params)
    data_loader = load_dataset.get_data_loader()

    learn_sigma = params.diffusion_mode == "vp_ddpm"
    ModelClass = PaperEEGDiffuser if arch_mode == "paper" else CodeEEGDiffuser

    model = ModelClass(
        in_channels=params.in_channels,
        time_points=params.time_points,
        num_classes=params.num_of_classes,
        learn_sigma=learn_sigma,
    )

    diffusion = create_diffusion(mode=params.diffusion_mode, timesteps=params.timesteps)
    trainer = Trainer(params=params, data_loader=data_loader, model=model, diffusion=diffusion)
    trainer.train()


if __name__ == "__main__":
    main()