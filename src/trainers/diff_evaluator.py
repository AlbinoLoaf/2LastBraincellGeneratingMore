import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset.datasets import LoadDataset
from diffusion import create_diffusion
from models.eegdiffuser import EEGDiffuser


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

            gen = torch.Generator(device=self.device)
            gen.manual_seed(i)
            t = torch.randint(
                0,
                self.diffusion.num_timesteps,
                (x.shape[0],),
                generator=gen,
                device=self.device,
            )
            model_kwargs = dict(y=y)
            loss_dict = self.diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            losses.append(loss.data.cpu().numpy())

        avg_val_loss = float(np.mean(losses))
        return avg_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EEGDiffuser")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to config file"
    )
    cmd_args = parser.parse_args()

    with open(cmd_args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    params = ConfigArgs(config_dict)

    if torch.cuda.is_available():
        torch.cuda.set_device(params.cuda)

    load_dataset = LoadDataset(params)
    data_loader = load_dataset.get_data_loader()

    diffusion = create_diffusion(timestep_respacing="")
    evaluator = Evaluator(
        params=params, data_loader=data_loader["val"], diffusion=diffusion
    )

    model = EEGDiffuser(num_classes=params.num_of_classes).to(evaluator.device)

    checkpoints = glob.glob(os.path.join(params.model_dir, "*.pth"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        model.load_state_dict(
            torch.load(latest_checkpoint, map_location=evaluator.device)
        )
    else:
        print("No checkpoint found in model_dir. Evaluating with initial weights.")

    with torch.no_grad():
        val_loss = evaluator.get_metrics(model)
        print(f"Validation Loss: {val_loss:.5f}")