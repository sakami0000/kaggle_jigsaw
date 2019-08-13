import torch
from torch import nn


class CustomLoss(nn.Module):

    def __init__(self, loss_weight=None, alpha=1, beta=1, use_annotator_counts=False,
                 weight_from_annotator_counts=None):
        super(CustomLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.use_annotator_counts = use_annotator_counts
        self.weight_from_annotator_counts = weight_from_annotator_counts

    def forward(self, logits, targets, annotator_counts=None):
        """
        preds[:, 0] = "prediction for target labels"
        preds[:, 1:] = "prediction for auxiliary target labels"
        targets[:, 0] = "target labels"
        targets[:, 1] = "instance weight"
        targets[:, 2:] = "auxiliary target labels"
        """
        if self.loss_weight is None:
            weight = None
            loss_weight = 1
        else:
            weight = targets[:, 1:2]
            loss_weight = self.loss_weight
    
        if annotator_counts is None or not self.use_annotator_counts:
            bce_loss_1 = nn.BCEWithLogitsLoss(weight=weight)(logits[:, :1], targets[:, :1])
            bce_loss_2 = nn.BCEWithLogitsLoss()(logits[:, 1:], targets[:, 2:])
            return (bce_loss_1 * loss_weight) + bce_loss_2
        else:
            annotator_counts = annotator_counts.view(-1, 1)
            new_targets = targets.clone()
            new_targets[:, :1] = (targets[:, :1] * annotator_counts + self.alpha) / (
                annotator_counts + self.alpha + self.beta)

            num_aux_targets = targets.size()[1] - 1
            aux_annotator_counts = annotator_counts.view(-1, 1).repeat(1, num_aux_targets)
            new_targets[:, 1:] = (targets[:, 1:] * aux_annotator_counts + self.alpha) / (
                aux_annotator_counts + self.alpha + self.beta)

            bce_loss_1 = nn.BCEWithLogitsLoss(weight=weight, reduction='none')(
                logits[:, :1], new_targets[:, :1])
            bce_loss_2 = torch.mean(nn.BCEWithLogitsLoss(reduction='none')(
                logits[:, 1:], new_targets[:, 2:]), 1).view(-1, 1)
            if self.weight_from_annotator_counts is None:
                return ((bce_loss_1 * loss_weight) + bce_loss_2).mean()
            return (((bce_loss_1 * loss_weight) + bce_loss_2) * self.weight_from_annotator_counts(
                annotator_counts + self.alpha + self.beta)).mean()
