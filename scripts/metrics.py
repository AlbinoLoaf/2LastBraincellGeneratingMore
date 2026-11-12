import argparse
import json
import yaml
import pathlib as p
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import utils.metrics_utils as metrics
from dataset.datasets import CustomDataset
from models.cbramod import CBraMod

class ConfigArgs:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def main():
    parser = argparse.ArgumentParser(description='CBraMod Evaluation Pipeline')
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file")
    cmd_args = parser.parse_args()
    
    # Load YAML config
    with open(cmd_args.config, 'r') as file:
        config_dict = yaml.safe_load(file)

    finetune_defaults = {
        'cuda': 0, 'parallel': False, 'epochs': 50, 'batch_size': 64,
        'lr': 1e-4, 'weight_decay': 5e-2, 'clip_value': 1, 'lr_scheduler': 'CosineAnnealingLR',
        'in_dim': 200, 'out_dim': 200, 'd_model': 200, 'dim_feedforward': 800,
        'seq_len': 10, 'n_layer': 12, 'nhead': 8, 'need_mask': False,
        'num_of_classes': 9,
        'model_dir': './models_weights/CBraMod_Finetuned',
        'datasets_dir': './datasets/real_data_lmdb'
    }

    finetune_defaults.update(config_dict)
    arch_params = ConfigArgs(finetune_defaults)
    
    print(f"--- STARTING EVALUATION PIPELINE ({arch_params.mode.upper() if hasattr(arch_params, 'mode') else 'UNKNOWN'} MODE) ---")

    model_dir = p.Path(arch_params.model_dir)
    save_dir = p.Path(f"./evaluation_results/{arch_params.mode}/") / model_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f"cuda:{arch_params.cuda}" if torch.cuda.is_available() else "cpu")
    
    # Process JSON history
    history_file = model_dir / 'training_history.json'
    metrics.plot_training_history(history_file, save_dir)

    test_dataset = CustomDataset(arch_params.datasets_dir, mode='test')
    dataloader = DataLoader(test_dataset, batch_size=arch_params.batch_size, shuffle=False)
    
    metrics.evaluate_and_save_all(
        model_dir=model_dir,
        dataloader=dataloader,
        model_class=CBraMod,
        arch_params=arch_params,
        save_dir=save_dir,
        device=device,
    )

if __name__ == "__main__":
    main()
