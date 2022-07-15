import torch


def weighted_loss(y_true, y_pred, pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss value.

    Args:
        y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
        y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
    Returns:
        loss (Tensor): overall scalar loss summed across all classes
    """
    # initialize loss to zero
    loss = 0.0

    for i in range(len(pos_weights)):
        # for each class, add average weighted loss for that class
        loss -= torch.mean(
            pos_weights[i] * y_true[:, i] * torch.log(
                y_pred[:, i] + epsilon) + \
            neg_weights[i] * (
                    1 - y_true[:, i]) * torch.log(
                1 - y_pred[:, i] + epsilon), axis=0)

    return loss
