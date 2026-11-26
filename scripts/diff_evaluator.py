import argparse
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import yaml

from dataset.datasets import LoadDataset
from diffusion import create_diffusion
from models.eegdiffuser import EEGDiffuser as CodeEEGDiffuser
from models.paper_eegdiffuser import EEGDiffuser as PaperEEGDiffuser


class ConfigArgs:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


class Evaluator:
    def __init__(self, params, data_loader, diffusion):
        self.params = params
        self.data_loader = data_loader
        self.diffusion = diffusion
        self.device = torch.device(
            f"cuda:{self.params.cuda}"
            if torch.cuda.is_available() and hasattr(self.params, "cuda")
            else "cpu"
        )

    def get_metrics(self, model):
        model.eval()

        losses = []
        for i, (x, y) in tqdm(enumerate(self.data_loader), mininterval=1):
            x = x.to(self.device)
            y = y.to(self.device)

            loss = self.diffusion.training_loss(model, x, y)

            losses.append(loss.data.cpu().numpy())

        avg_val_loss = float(np.mean(losses))
        return avg_val_loss

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EEGDiffuser")
    
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config file")
    cmd_args = parser.parse_args()

    with open(cmd_args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    params = ConfigArgs(config_dict)

    if torch.cuda.is_available():
        torch.cuda.set_device(params.cuda)

    load_dataset = LoadDataset(params)
    data_loader = load_dataset.get_data_loader()

    
    diffusion_mode = getattr(params, "diffusion_mode", "vp_ddpm")
    timesteps = getattr(params, "timesteps", 1000)
    arch_mode = getattr(params, "arch_mode", "paper")
    learn_sigma = diffusion_mode == "vp_ddpm"

    diffusion = create_diffusion(mode=diffusion_mode, timesteps=timesteps)
    evaluator = Evaluator(
        params=params, data_loader=data_loader["val"], diffusion=diffusion
    )


    ModelClass = PaperEEGDiffuser if arch_mode == "paper" else CodeEEGDiffuser
    model = ModelClass(
        in_channels=params.in_channels,
        time_points=params.time_points,
        num_classes=params.num_of_classes,
        learn_sigma=learn_sigma,
    ).to(evaluator.device)


    best_ckpt = os.path.join(params.model_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt_path = best_ckpt
    else:
        checkpoints = glob.glob(os.path.join(params.model_dir, "*.pth"))
        ckpt_path = max(checkpoints, key=os.path.getmtime) if checkpoints else None

    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=evaluator.device))
    else:
        print("No checkpoint found in model_dir. Evaluating with initial weights.")

    with torch.no_grad():
        val_loss = evaluator.get_metrics(model)
        print(f"Validation Loss ({arch_mode.upper()} | {diffusion_mode.upper()}): {val_loss:.5f}")