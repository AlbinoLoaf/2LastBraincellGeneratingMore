import argparse
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset

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

def main():
    parser = argparse.ArgumentParser(description='CBraMod Fine-Tuning')
    parser.add_argument('--config', type=str, default='config.yaml')
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as file:
        config_dict = yaml.safe_load(file)


    finetune_defaults = {
        'cuda': 0, 'parallel': False, 'epochs': 50, 'batch_size': 64,
        'lr': 1e-4, 'weight_decay': 5e-2, 'clip_value': 1, 'lr_scheduler': 'CosineAnnealingLR',
        'in_dim': 200, 'out_dim': 200, 'd_model': 200, 'dim_feedforward': 800,
        'seq_len': 10, 'n_layer': 12, 'nhead': 8, 'need_mask': False, 
        'num_of_classes': 9,
        'pretrained_weights': './models_weights/CBraMod_Pretrain/pretrained_weights.pth',
        'model_dir': './models_weights/CBraMod_Finetuned'
    }

    finetune_defaults.update(config_dict)
    params = ConfigArgs(finetune_defaults)
    setup_seed(params.seed)
    print("Loading Real only")
    real_train_dataset = CustomDataset(params.datasets_dir, mode='train')
    # syn_train_dataset = CustomDataset(params.synthetic_data_dir, mode='train')
    # mixed_train_dataset = ConcatDataset([real_train_dataset, syn_train_dataset])
    
    # Train exclusively on real data
    train_loader = DataLoader(real_train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)
    val_dataset = CustomDataset(params.datasets_dir, mode='val')
    test_dataset = CustomDataset(params.datasets_dir, mode='test')
    
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    print("Initializing Model and Loading Pretrained Weights...")
    
    model = CBraMod(
        params.in_dim, params.num_of_classes, params.d_model, params.dim_feedforward,
        params.seq_len, params.n_layer, params.nhead
    )
    try:
        pretrained_dict = torch.load(params.pretrained_weights, map_location='cpu')
        model.load_state_dict(pretrained_dict, strict=False)
        print("Pretrained weights loaded successfully.")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}. Starting from scratch.")
    trainer = Trainer(params, data_loaders, model)
    trainer.train()

if __name__ == '__main__':
    main()

