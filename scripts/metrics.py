import pathlib as p
import torch
from torch.utils.data import DataLoader

import utils.metrics_utils as metrics_utils
from dataset.datasets import CustomDataset
from models.cbramod import CBraMod

def main():
    print("Starting evaluation pipeline...")
    model_dir = p.Path("./models_weights/CBraMod_Finetuned")
    save_dir = p.Path("./evaluation_results")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arch_params = {
        "in_dim": 200,
        "num_classes": 9,
        "d_model": 200,
        "dim_feedforward": 800,
        "seq_len": 10,
        "n_layer": 12,
        "nhead": 8
    }
    dataset_dir = "./datasets/real_data_lmdb"
    test_dataset = CustomDataset(dataset_dir, mode='test')
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    metrics_utils.evaluate_and_save_all(
        model_dir=model_dir,
        dataloader=dataloader,
        model_class=CBraMod,
        arch_params=arch_params,
        save_dir=save_dir,
        device=device,
    )

if __name__ == "__main__":
    main()