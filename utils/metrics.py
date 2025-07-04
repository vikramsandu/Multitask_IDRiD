import torch
import torch.nn as nn


# Compute Dice Score.
def compute_dice(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


# Learnable Uncertainty Weighted Loss
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))

    def forward(self, cls_loss, seg_loss):
        weighted_cls = torch.exp(-self.log_var_cls) * cls_loss + self.log_var_cls
        weighted_seg = torch.exp(-self.log_var_seg) * seg_loss + self.log_var_seg
        return weighted_cls + weighted_seg, weighted_cls.item(), weighted_seg.item()
