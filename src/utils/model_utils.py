import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset.datasets import CustomDataset
from models.cbramod import CBraMod
from finetune_trainer import Trainer

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_dataloaders(data_mode: str, datasets_dir: str, synthetic_data_dir: str, batch_size: int, num_workers: int = 2):
    print(f"Loading data in [{data_mode.upper()}] mode...")
    
    if data_mode == 'real_only':
        train_dataset = CustomDataset(datasets_dir, mode='train')
    elif data_mode == 'synth_only':
        train_dataset = CustomDataset(synthetic_data_dir, mode='train')
    elif data_mode == 'mixed':
        real_train = CustomDataset(datasets_dir, mode='train')
        syn_train = CustomDataset(synthetic_data_dir, mode='train')
        train_dataset = ConcatDataset([real_train, syn_train])
    else:
        raise ValueError("Invalid data_mode. Use 'real_only', 'mixed', or 'synth_only'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = CustomDataset(datasets_dir, mode='val')
    test_dataset = CustomDataset(datasets_dir, mode='test')
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def load_pretrained_weights(model, weight_path, strict=False):
    try:
        pretrained_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(pretrained_dict, strict=strict)
        print(f"Pretrained weights successfully loaded from {weight_path}")
    except Exception as e:
        print(f"Warning: Could not load weights from {weight_path}. Error: {e}")
        print("Model will be randomly initialized.")
    return model

def get_predictions(model, data_loader, device):
    """
    Extracts true labels and model predictions over a dataset.
    Handles the 4D to 2D pooling required for CBraMod classification.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, labels in data_loader:
            x = x.float().to(device) / 100.0
            labels = labels.long().to(device)
            
            # Reshape input to (Batch, Channels, Patches, Points)
            batch_size = x.shape[0]
            x = x.view(batch_size, 32, 10, 200)
            
            outputs = model(x)
            
            # Pool spatial/temporal dimensions if output is 4D
            if outputs.dim() == 4:
                outputs = outputs.mean(dim=(1, 2))
                
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def calculate_classification_metrics(y_true, y_pred):
    """
    Calculates the specific metrics required for the FACED dataset evaluation.
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'balanced_accuracy': bal_acc,
        'cohen_kappa': kappa,
        'weighted_f1': f1
    }

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """
    Generates and optionally saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close(fig)

def load_pretrained_weights(model, weight_path, strict=False):
    """
    Safely loads pretrained weights, handling missing classification heads.
    """
    try:
        pretrained_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        model.load_state_dict(pretrained_dict, strict=strict)
        print(f"Pretrained weights successfully loaded from {weight_path}")
    except Exception as e:
        print(f"Warning: Could not load weights from {weight_path}. Error: {e}")
        print("Model will be randomly initialized.")
    return model