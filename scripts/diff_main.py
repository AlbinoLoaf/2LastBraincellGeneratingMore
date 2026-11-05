import argparse
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset
import subprocess
import sys

from dataset.datasets import CustomDataset
from models.cbramod import CBraMod
from finetune_trainer import Trainer

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

def get_dataloaders(params):
    print(f"Loading data in [{params.data_mode.upper()}] mode...")
    
    # 1. Build the Training Set based on data_mode config
    if params.data_mode == 'real_only':
        train_dataset = CustomDataset(params.datasets_dir, mode='train')
    
    elif params.data_mode == 'synth_only':
        train_dataset = CustomDataset(params.synthetic_data_dir, mode='train')
        
    elif params.data_mode == 'mixed':
        real_train = CustomDataset(params.datasets_dir, mode='train')
        syn_train = CustomDataset(params.synthetic_data_dir, mode='train')
        train_dataset = ConcatDataset([real_train, syn_train])
        
    else:
        raise ValueError("Invalid data_mode. Use 'real_only', 'mixed', or 'synth_only'.")

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)

    # 2. Validation and Test sets are ALWAYS strictly real data
    val_dataset = CustomDataset(params.datasets_dir, mode='val')
    test_dataset = CustomDataset(params.datasets_dir, mode='test')
    
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def run_finetuning(params):
    data_loaders = get_dataloaders(params)
    
    print("Initializing Model for Fine-Tuning...")
    model = CBraMod(
        params.in_dim, params.num_of_classes, params.d_model, params.dim_feedforward,
        params.seq_len, params.n_layer, params.nhead
    )
    
    if hasattr(params, 'pretrained_weights') and params.pretrained_weights:
        try:
            pretrained_dict = torch.load(params.pretrained_weights, map_location='cpu')
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"Pretrained weights loaded from {params.pretrained_weights}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}. Starting from scratch.")
            
    trainer = Trainer(params, data_loaders, model)
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description='Universal Pipeline Controller')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Base CBraMod defaults to ensure architecture builds correctly
    defaults = {
        'task': 'finetune', 'data_mode': 'real_only', 'cuda': 0, 'parallel': False, 
        'epochs': 50, 'batch_size': 32, 'lr': 1e-4, 'weight_decay': 5e-2, 
        'clip_value': 1, 'lr_scheduler': 'CosineAnnealingLR', 'in_dim': 200, 
        'd_model': 200, 'dim_feedforward': 800, 'seq_len': 10, 'n_layer': 12, 'nhead': 8
    }

    defaults.update(config_dict)
    params = ConfigArgs(defaults)
    setup_seed(params.seed)

    print(f"--- Starting Task: {params.task.upper()} ---")

    if params.task == 'finetune':
        run_finetuning(params)
        
    elif params.task == 'pretrain':
        # Route to your pretraining logic (can be imported or called directly)
        print("Executing pretraining pipeline...")
        subprocess.run([sys.executable, "scripts/pretrain_main.py", "--config", cmd_args.config])
        
    elif params.task == 'train_diffusion':
        # Route to your EEGDiffuser training logic
        print("Executing EEGDiffuser training pipeline...")
        subprocess.run([sys.executable, "scripts/train_diffusion.py", "--config", cmd_args.config])
        
    else:
        print(f"Task '{params.task}' not recognized.")

if __name__ == '__main__':
    main()