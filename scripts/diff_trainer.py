import argparse
import glob
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from timeit import default_timer as timer
from collections import OrderedDict
from copy import deepcopy
import lmdb
import pickle
from tqdm import tqdm

from dataset.datasets import LoadDataset
from diff_evaluator import Evaluator
from utils.util import VLBLoss, draw
from diffusion import create_diffusion
from models.eegdiffuser import EEGDiffuser


class ConfigArgs:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


class Trainer(object):
    def __init__(self, params, data_loader, model, diffusion=None):
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(
            f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu"
            )
        self.model = model.to(self.device)

        self.criterion = VLBLoss().to(self.device)

        if self.params.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay
            )
        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

#        self.diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        diffusion_mode = getattr(self.params, "diffusion_mode")
        timesteps = getattr(self.params, "timesteps", 1000)
        self.diffusion = create_diffusion(mode=diffusion_mode, timesteps=timesteps)
        self.evaluator = Evaluator(params, self.data_loader['val'], self.diffusion)

    def train(self):
        ema = deepcopy(self.model).to(self.device)
        requires_grad(ema, False)

        update_ema(ema, self.model, decay=0)
        self.model.train()
        ema.eval()

        os.makedirs(self.params.model_dir, exist_ok=True)
        print(f"Training for {self.params.epochs} epochs...")
        best_avg_loss = float("inf")
        best_epoch = 0
        best_model_path = os.path.join(self.params.model_dir, "best_model.pth")
        grad_clip = getattr(self.params, "grad_clip", 1.0)

        for epoch in range(self.params.epochs):
            print(f"Beginning epoch {epoch}...")
            start_time = timer()
            losses = []
            self.model.train()

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                loss = self.diffusion.training_loss(self.model, x, y)
                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()
                update_ema(ema, self.model)
                self.optimizer_scheduler.step()

                losses.append(loss.item())

            avg_training_loss = np.mean(losses)
            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                avg_val_loss = self.evaluator.get_metrics(model=ema)
                print(
                    "Epoch {} : Training Loss: {:.5f}, Validation Loss: {:.5f}, LR: {:.6f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        avg_training_loss,
                        avg_val_loss,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if best_avg_loss > avg_val_loss:
                    best_epoch = epoch + 1
                    best_avg_loss = avg_val_loss
                    model_path = os.path.join(self.params.model_dir, f"epoch{epoch + 1}_avgloss_{avg_val_loss:.5f}.pth")
                    torch.save(ema.state_dict(), model_path)
                    torch.save(ema.state_dict(), best_model_path)
                    print("model saved in " + model_path)

            if epoch + 1 == self.params.epochs:
                print("{} epoch got the best avgloss {:.5f}".format(best_epoch, best_avg_loss))
                print("best model saved in " + best_model_path)

        return best_avg_loss

    def _load_latest_model(self):
        best_path = os.path.join(self.params.model_dir, "best_model.pth")
        if os.path.exists(best_path):
            ckpt_path = best_path
        else:
            checkpoints = glob.glob(os.path.join(self.params.model_dir, "*.pth"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint files found in {self.params.model_dir}")
            ckpt_path = max(checkpoints, key=os.path.getmtime)

        print(f"Loading weights from: {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def sample(self):
        CHANNEL_LIST = [
            'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8',
            'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4',
            'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6',
            'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4',
            'Oz', 'O1', 'O2', 'A2', 'A1'
        ]
        self._load_best_model()

        class_labels = [1, 1, 1, 4, 4, 4]
        y = torch.tensor(class_labels, dtype=torch.long, device=self.device)
        shape = (self.params.in_channels, self.params.time_points)

        samples = self.diffusion.sample_with_cfg(
            model=self.model,
            labels=y,
            num_classes=self.params.num_of_classes,
            shape=shape,
            cfg_scale=self.params.cfg_scale,
        )

        for sample in samples:
            draw(sample.cpu().numpy(), CHANNEL_LIST)

    def synthetic_data(self):
        self._load_latest_model()

        os.makedirs(self.params.synthetic_data_dir, exist_ok=True)
        db = lmdb.open(self.params.synthetic_data_dir, map_size=66125001720)
        test_n = 0
        keys = []

        shape = (self.params.in_channels, self.params.time_points)
        synthetic_ratio = getattr(self.params, "synthetic_ratio", 1)

        for epoch in range(synthetic_ratio):
            print(f'Generating synthetic pass {epoch + 1}/{synthetic_ratio}...')
            for _, y in tqdm(self.data_loader['train'], mininterval=10):
                y = y.long().to(self.device)

                samples = self.diffusion.sample_with_cfg(
                    model=self.model,
                    labels=y,
                    num_classes=self.params.num_of_classes,
                    shape=shape,
                    cfg_scale=self.params.cfg_scale,
                )

                for sample, label in zip(samples, y):
                    sample_arr = sample.contiguous().view(32, 10, 200).cpu().numpy()
                    label_arr = label.cpu().numpy()
                    data_dict = {'sample': sample_arr, 'label': label_arr}

                    txn = db.begin(write=True)
                    txn.put(key=str(test_n).encode(), value=pickle.dumps(data_dict))
                    txn.commit()
                    keys.append(str(test_n))
                    test_n += 1

        txn = db.begin(write=True)
        txn.put(key='__keys__'.encode(), value=pickle.dumps(keys))
        txn.commit()
        db.close()
        print(f'Saved {test_n} synthetic samples to {self.params.synthetic_data_dir}')

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EEGDiffuser')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    args = ConfigArgs(config_dict)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

    load_dataset = LoadDataset(args)
    data_loader = load_dataset.get_data_loader()

    model = EEGDiffuser(num_classes=args.num_of_classes)

    trainer = Trainer(params=args, data_loader=data_loader, model=model)
    #trainer.train()
    trainer.synthetic_data()
