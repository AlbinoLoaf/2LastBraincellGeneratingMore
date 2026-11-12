import os
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, ConfusionMatrixDisplay
# Functions for evaluating the performance of the fine-tuned models.
# The metrics consist of Balanced Accuracy, Cohen's Kappa, Weighted F1, Nearest Neighbour,
# and Differential Privacy.

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def plot_training_history(json_path, save_dir):
    if not json_path.exists():
        print(f"No training history found at {json_path}")
        return

    with open(json_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss curves')
    plt.legend()
    plt.savefig(save_dir / 'loss_curve.png')
    plt.close()
    print(f"Saved loss curve to {save_dir / 'loss_curve.png'}")

# Returns: Dict: Balanced Accuracy, Cohen's Kappa, Weighted F1
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns the balanced accuracy, Choens kappar, and weighted_f1 score of a set of true and pred labels
      
    Parameters
    -----------
    y_true : ndarray
        An array containing the ground truth for each input.
    y_pred : ndarray  
        An array containing the prediction for each input

    Return
    -----------
    metrics : dict{str:float}  
        A dictionary from balanced_accuracy, cohen_kappa, and weighted_f1 to their respective values.
    """

    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",zero_division=0)
    }

def get_preds_and_labels(model, dataloader, device):
    """Executes forward passes over a dataloader to extract true and predicted labels."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.float().to(device), y.long().to(device)
            
            # Auto-reshape for CBraMod if fed raw sequential data
            if x.dim() == 3 and x.shape[-1] == 2000:
                x = x.view(x.shape[0], 32, 10, 200)
                
            outputs = model(x)
            
            if outputs.dim() == 4:
                outputs = outputs.mean(dim=(1, 2))
                
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def load_all_models(folder_path, model_class, device, params):
    """Loads all models using explicit instantiation from the params object."""
    models = []
    folder = Path(folder_path)
    
    if not folder.exists():
        return models

    for file_path in folder.iterdir():
        if file_path.suffix == ".pth":
            
            model = model_class(
                params.in_dim, 
                params.num_of_classes, 
                params.d_model, 
                params.dim_feedforward,
                params.seq_len, 
                params.n_layer, 
                params.nhead
            ).to(device)
            
            model.load_state_dict(torch.load(file_path, map_location=device))
            missing, unexpected = model.load_state_dict(torch.load(file_path,map_location=device), strict=False)
            print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
            model.eval()
            models.append((file_path.name, model))
            
    return models

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Generates and saves a standardized confusion matrix."""
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def evaluate_and_save_all(model_dir, dataloader, model_class,
                          arch_params, save_dir, device):
    """
    The Stitcher: Loads all models, computes metrics, saves graphs, and exports a unified JSON report.
    """
    os.makedirs(save_dir, exist_ok=True)
    models = load_all_models(model_dir, model_class, device, arch_params)
    
    if not models:
        print(f"No models found in {model_dir}")
        return

    results = {}
    print(f"Evaluating {len(models)} models from {model_dir}...")
    
    for filename, model in models:
        print(f"Processing {filename}...")
        y_true, y_pred = get_preds_and_labels(model, dataloader, device)
        
        # Calculate numerical metrics
        metrics = classification_metrics(y_true, y_pred)
        results[filename] = metrics
        
        # Generate and save visual plots
        # Generate and save visual plots
        plot_name = os.path.join(save_dir, f"{filename.replace('.pth', '')}_confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, title=f"CM: {filename}", save_path=plot_name)
        
    # Save unified JSON report
    report_path = os.path.join(save_dir, "evaluation_metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Evaluation complete. All reports and graphs saved to {save_dir}")

def calculate_losses():
    
    raise NotImplementedError

def membership_inference_attack():
    raise NotImplementedError
 
def plot_membership_inference():
    raise NotImplementedError

#Calculates the nearest-neighbour distance between synthetic samples and real 
# training samples. Used to check for memorization.
# Edge cases: missing labels or empty distance lists: not handled
# Returns: Dictionary containing distance statistics.
def nearest_neighbours(training_data, synthetic_data, test_data, training_labels=None,
    synthetic_labels=None, test_labels=None):
    # Consider detatching
    #training_data = training_data.detach().cpu().numpy()

    training_data = to_numpy(training_data)
    synthetic_data = to_numpy(synthetic_data)
    test_data = to_numpy(test_data)

    # Flatten the data
    training_data = training_data.reshape(training_data.shape[0], -1)
    synthetic_data = synthetic_data.reshape(synthetic_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)

    # Class-conditioned comparison
    class_conditioned = training_labels is not None and synthetic_labels is not None and test_labels is not None

    if class_conditioned:
        training_labels = np.asarray(training_labels)
        synthetic_labels = np.asarray(synthetic_labels)
        test_labels = np.asarray(test_labels)

        synthetic_distances = []
        test_distances = []

        for label in np.unique(training_labels):
            real_class = training_data[training_labels == label]
            synthetic_class = synthetic_data[synthetic_labels == label]
            test_class = test_data[test_labels == label]

            if len(real_class) == 0:
                continue

            if len(synthetic_class) > 0:
                distances = pairwise_distances(
                    synthetic_class,
                    real_class,
                    metric="euclidean"
                )
                synthetic_distances.extend(distances.min(axis=1))

            if len(test_class) > 0:
                distances = pairwise_distances(
                    test_class,
                    real_class,
                    metric="euclidean"
                )
                test_distances.extend(distances.min(axis=1))
    else:
        synthetic_distances = pairwise_distances(
            synthetic_data,
            training_data,
            metric="euclidean"
        ).min(axis=1)

        test_distances = pairwise_distances(
            test_data,
            training_data,
            metric="euclidean"
        ).min(axis=1)

    return {
        "class_conditioned": class_conditioned,
        "synthetic_mean_nnd": np.mean(synthetic_distances),
        "synthetic_median_nnd": np.median(synthetic_distances),
        "test_mean_nnd": np.mean(test_distances),
        "test_median_nnd": np.median(test_distances)
    }

if __name__ == "__main__":
    0

# samlet function -> main
# make figures like the ones in the paper
# make script that runs this and saves things in the appropriate folders
