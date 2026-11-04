import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, pairwise_distances, roc_auc_score

# Functions for evaluating the performance of the fine-tuned models.
# The metrics consist of Balanced Accuracy, Cohen's Kappa, Weighted F1, Nearest Neighbour,
# and Differential Privacy.

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)

# Returns: Dict: Balanced Accuracy, Cohen's Kappa, Weighted F1
def classification_metrics(y_true, y_pred):
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",zero_division=0)
    }
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