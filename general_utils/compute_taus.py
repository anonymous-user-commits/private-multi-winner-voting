import numpy as np
from sklearn import metrics


def compute_taus_per_label(votes, targets):
    num_labels = votes.shape[1]
    taus = []
    weights = []
    for label in range(num_labels):
        votes_label = votes[:, label]
        targets_label = targets[:, label]
        no_nans = ~np.isnan(targets_label)
        votes_label = votes_label[no_nans]
        targets_label = targets_label[no_nans]
        if len(votes_label) > 0 and len(targets_label) > 0:
            ba_max = 0.0
            tau_best = 0.0
            for tau in np.linspace(0, 1, 100):
                preds = np.copy(votes_label)
                preds[preds > tau] = 1
                preds[preds <= tau] = 0
                balanced_acc = metrics.balanced_accuracy_score(
                    y_true=targets_label, y_pred=preds)
                print(label, ',', tau, ',', balanced_acc)
                if balanced_acc > ba_max:
                    tau_best = tau
                    ba_max = balanced_acc
            print(label, ',', tau_best, ',', ba_max)
            taus.append(tau_best)
            weights.append(ba_max)
        else:
            taus.append(np.nan)
            weights.append(np.nan)
    return taus, weights
