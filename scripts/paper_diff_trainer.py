import argparse
import glob
import os
import pickle
import lmdb
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from dataset.datasets import LoadDataset
from diffusion import create_diffusion
from models.eegdiffuser import EEGDiffuser


class ConfigArgs:
    def __init__(self, dictionary: dict):
        for k, v in dictionary.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, params: ConfigArgs, data_loader: dict, model: nn.Module):
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(f"cuda:{params.cuda}" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.diffusion = create_diffusion(mode=params.diffusion_mode, timesteps=params.timesteps).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay
        )
        total_steps = params.epochs * len(data_loader["train"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=params.min_lr
        )

    @torch.no_grad()
    def evaluate_val_loss(self) -> float:
        self.model.eval()
        val_losses = []
        for x, y in self.data_loader["val"]:
            x, y = x.to(self.device), y.to(self.device)
            val_losses.append(self.diffusion.training_loss(self.model, x, y).item())
        self.model.train()
        return float(np.mean(val_losses))

    def train(self):
        os.makedirs(self.params.model_dir, exist_ok=True)
        best_val_loss = float("inf")
        best_path = ""

        self.model.train()
        for epoch in range(self.params.epochs):
            train_losses = []
            for x, y in tqdm(self.data_loader["train"], desc=f"Epoch {epoch + 1}/{self.params.epochs}", mininterval=10):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                loss = self.diffusion.training_loss(self.model, x, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.params.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                train_losses.append(loss.item())

            val_loss = self.evaluate_val_loss()
            print(f"Epoch {epoch + 1} | Train Loss: {np.mean(train_losses):.5f} | Val Loss: {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(self.params.model_dir, f"best_epoch{epoch + 1}_val{val_loss:.5f}.pth")
                torch.save(self.model.state_dict(), best_path)

        print(f"Training complete. Best checkpoint: {best_path} (Val Loss: {best_val_loss:.5f})")

    def _load_best_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.params.model_dir, "*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.params.model_dir}")
        best_ckpt = max(checkpoints, key=os.path.getmtime)
        print(f"Loading checkpoint: {best_ckpt}")
        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
        self.model.eval()

    def generate_synthetic_lmdb(self):
        """Synthesize a 1:1 class-matched dataset in LMDB format for downstream CBraMod fine-tuning."""
        self._load_best_checkpoint()
        os.makedirs(os.path.dirname(self.params.synthetic_data_dir), exist_ok=True)
        db = lmdb.open(self.params.synthetic_data_dir, map_size=66125001720)

        idx = 0
        keys = []
        eeg_shape = (self.params.in_channels, self.params.time_points)

        for _, y in tqdm(self.data_loader["train"], desc="Synthesizing 1:1 EEG"):
            y = y.to(self.device)
            samples = self.diffusion.sample_with_cfg(
                model=self.model,
                labels=y,
                num_classes=self.params.num_of_classes,
                shape=eeg_shape,
                cfg_scale=self.params.cfg_scale,
            )
            for sample, label in zip(samples, y):
                # Reshape (32, 2000) -> (32, 10, 200) matching CBraMod's 1-second patch expectation
                sample_np = sample.contiguous().view(self.params.in_channels, 10, 200).cpu().numpy()
                label_np = label.cpu().numpy()

                with db.begin(write=True) as txn:
                    txn.put(key=str(idx).encode(), value=pickle.dumps({"sample": sample_np, "label": label_np}))
                keys.append(str(idx))
                idx += 1

        with db.begin(write=True) as txn:
            txn.put(key="__keys__".encode(), value=pickle.dumps(keys))
        db.close()
        print(f"Saved {idx} synthetic trials to {self.params.synthetic_data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEGDiffuser Replication Trainer")
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument("--mode", type=str, choices=["train", "synthesize"], default="train")
    cli_args = parser.parse_args()

    with open(cli_args.config, "r") as f:
        args = ConfigArgs(yaml.safe_load(f))

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

    data_loader = LoadDataset(args).get_data_loader()
    model = EEGDiffuser(
        in_channels=args.in_channels,
        time_points=args.time_points,
        num_classes=args.num_of_classes,
        learn_sigma=(args.diffusion_mode == "vp_ddpm"),
    )

    trainer = Trainer(params=args, data_loader=data_loader, model=model)
    if cli_args.mode == "train":
        trainer.train()
    else:
        trainer.generate_synthetic_lmdb()