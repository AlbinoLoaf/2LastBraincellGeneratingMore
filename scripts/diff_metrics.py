import argparse
import yaml
import torch
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from dataset.datasets import CustomDataset
from tqdm import tqdm

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    return (mu1 - mu2).dot(mu1 - mu2) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def main():
    parser = argparse.ArgumentParser(description='Measure Diffusion Quality')
    parser.add_argument('--config', type=str, required=True)
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as file:
        config = yaml.safe_load(file)

    print("Loading datasets for distribution comparison...")
    real_ds = CustomDataset(config['datasets_dir'], mode='train')
    syn_ds = CustomDataset(config['synthetic_data_dir'], mode='train')

    # Extract all tensors and flatten them from [32, 10, 200] to [64000]
    real_data = torch.stack([real_ds[i][0] for i in tqdm(range(len(real_ds)), desc="Real")]).view(len(real_ds), -1).numpy()
    syn_data = torch.stack([syn_ds[i][0] for i in tqdm(range(len(syn_ds)), desc="Synth")]).view(len(syn_ds), -1).numpy()

    print("Calculating Mean and Covariance...")
    mu_real, sigma_real = np.mean(real_data, axis=0), np.cov(real_data, rowvar=False)
    mu_syn, sigma_syn = np.mean(syn_data, axis=0), np.cov(syn_data, rowvar=False)

    print("Calculating Fréchet Distance...")
    fd = calculate_frechet_distance(mu_real, sigma_real, mu_syn, sigma_syn)
    
    print("Calculating average marginal Wasserstein Distance...")
    w_dists = [wasserstein_distance(real_data[:, i], syn_data[:, i]) for i in tqdm(range(real_data.shape[1]), desc="Wasserstein")]
    avg_wd = np.mean(w_dists)

    print("\n--- DIFFUSION QUALITY METRICS ---")
    print(f"Fréchet Distance: {fd:.4f} (Lower is better, >100 is severe domain shift)")
    print(f"Avg Wasserstein Distance: {avg_wd:.6f} (Lower is better)")

if __name__ == "__main__":
    main()
