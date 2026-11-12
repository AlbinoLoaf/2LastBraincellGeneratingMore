import argparse
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.datasets import LoadDataset
from models.cbramod import CBraMod
from pretrain_trainer import Trainer

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    cmd_args = parser.parse_args()


    with open(cmd_args.config, 'r') as file:
        config_dict = yaml.safe_load(file)

    cbramod_defaults = {
        'cuda': 0, 'parallel': False, 'epochs': 40, 'batch_size': 128,
        'lr': 5e-4, 'weight_decay': 5e-2, 'clip_value': 1, 'lr_scheduler': 'CosineAnnealingLR',
        'in_dim': 200, 'out_dim': 200, 'd_model': 200, 'dim_feedforward': 800,
        'seq_len': 10, 'n_layer': 12, 'nhead': 8, 'need_mask': True, 'mask_ratio': 0.5,
        'model_dir': './models_weights/CBraMod_Pretrain'
    }

    cbramod_defaults.update(config_dict)
    params = ConfigArgs(cbramod_defaults)
    
    setup_seed(params.seed)
    load_dataset = LoadDataset(params)
    data_loader = load_dataset.get_data_loader()['train']

    model = CBraMod(
        params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, 
        params.seq_len, params.n_layer, params.nhead
    )
    
    trainer = Trainer(params, data_loader, model)
    trainer.train()

if __name__ == '__main__':
    main()
