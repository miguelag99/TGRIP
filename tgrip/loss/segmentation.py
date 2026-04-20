import pdb
from functools import partial
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from torchvision.ops import sigmoid_focal_loss

from tgrip.loss import LossInterface


def select_loss(loss_fn, pred, target, time_index, scale_index, channel_index):
    if time_index is not None:
        loss = loss_fn(pred[:, time_index], target[:, time_index]).unsqueeze(1)
    elif scale_index is not None:
        loss = loss_fn(pred[scale_index], target[scale_index]).unsqueeze(0)
    elif channel_index is not None:
        loss = loss_fn(
            pred[:, :, channel_index], target[:, :, channel_index]
        ).unsqueeze(2)
    else:
        loss = loss_fn(pred, target)
    return loss


# Binimg losses.
class BCELoss(LossInterface):
    def __init__(
        self,
        pos_weight,
        key="binimg",
        name="loss_binimg",
        time_index: Optional[int] = None,
        scale_index: Optional[int] = None,
        channel_index: Optional[int] = None,
        select_index: Optional[int] = False,
    ):
        """
        BCE(p) = -(y * log(p) + (1 - y) * log(1 - p))

        if y=0:
            BCE(p) = -log(1 - p)
            - if p ~ 0:
                well classified and BCE(p) ~ 0
            - if p ~ 1:
                badly classified and BCE(p) ~ inf

        if y=1:
            BCE(p) = -log(p)
            - if p ~ 0:
                badly classified and BCE(p) ~ inf
            - if p ~ 1:
                well classified and BCE(p) ~ 0
        """

        super().__init__(key=key, name=name)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]), reduction="none"
        )
        self.time_index = time_index
        self.scale_index = scale_index
        self.channel_index = channel_index
        self.select_index = select_index
        assert not (
            int(self.time_index is not None)
            + int(self.scale_index is not None)
            + int(self.channel_index is not None)
            > 1
        )

    def forward(self, pred, target, mask=None, target_weights=None, eps=1e-6):
        loss = select_loss(
            self.loss_fn,
            pred,
            target,
            self.time_index,
            self.scale_index,
            self.channel_index,
        )

        if target_weights is not None:
            loss = loss * (target * target_weights + (1 - target))

        if mask is None:
            mask = torch.ones_like(loss, dtype=torch.bool)

        return (loss * mask).sum() / (mask.sum() + eps)


class SpatialLoss(LossInterface):
    def __init__(self, norm, key="offsets", name="loss_offsets", ignore_index=None):
        super().__init__(key=key, name=name)

        if norm == 1:
            self.loss_fn = torch.nn.functional.l1_loss
        elif norm == 2:
            self.loss_fn = torch.nn.functional.mse_loss
        elif norm == 1.5:
            self.loss_fn = torch.nn.functional.smooth_l1_loss
        else:
            raise NotImplementedError
        self.ignore_index = ignore_index

    def forward(self, pred, target, mask=None, eps=1e-6) -> torch.Tensor:
        # Alias
        b, t, c, h, w = pred.shape
        loss = self.loss_fn(pred, target, reduction="none")

        if self.ignore_index is not None:
            target_mask = target != self.ignore_index
        else:
            target_mask = torch.ones_like(loss, dtype=torch.bool)

        if mask is None:
            mask = torch.ones_like(loss, dtype=torch.bool)

        mask = target_mask & mask
        return (loss * mask).sum() / (mask.sum() + eps)


## PowerBEV segmentation loss

class CELoss(nn.Module):
    def __init__(
        self,
        class_weights,
        ignore_index=255,
        use_top_k=False,
        top_k_ratio=1.0,
        future_discount=1.0
    ):
        super().__init__()
        self.class_weights = torch.Tensor(class_weights)
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount
        
    def forward(self, prediction, target, mask=None, eps=1e-6):
        if target.shape[-3] != 1:
            raise ValueError('segmentation label must be an index-label with channel dimension = 1.')
        b, s, c, h, w = prediction.shape
        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w).long()
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )
        
        loss = loss.view(b, s, h, w)

        future_discounts = self.future_discount ** torch.arange(s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)

class FocalLoss(nn.Module):
    def __init__(
        self,
        class_weights=[0.1, 0.4, 0.8],
        gamma=2.0,
        ignore_index=255,
        future_discount=0.95,
        reduction="mean",
    ):
        """
        Focal Loss for BEV forecasting with temporal discounting.

        Args:
            class_weights (list): Static class weights. E.g., [0.1, 0.4, 0.8]
            gamma (float): Focusing parameter for hard examples.
            ignore_index (int): Index to ignore (pixels outside the map).
            future_discount (float): Discount factor in [0, 1] to penalize future frames less.
            reduction (str): 'mean', 'sum' or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.future_discount = future_discount
        self.reduction = reduction
        
        if isinstance(class_weights, list):
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        elif isinstance(class_weights, torch.Tensor):
            self.class_weights = class_weights.float()
        else:
            self.class_weights = None

    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: Predictions [B, T, C, H, W]
            targets: Ground truth [B, T, 1, H, W] (values from 0 to C-1)
        """
        if self.class_weights is not None and self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)

        inputs = inputs.transpose(1, 2)  # [B, C, T, H, W]
        targets = targets.squeeze(2).long()  # [B, T, H, W]

        ce_loss_unweighted = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index
        )
        
        pt = torch.exp(-ce_loss_unweighted)

        if self.class_weights is not None:
            ce_loss_weighted = F.cross_entropy(
                inputs,
                targets,
                weight=self.class_weights,
                reduction="none",
                ignore_index=self.ignore_index,
            )
        else:
            ce_loss_weighted = ce_loss_unweighted

        # Output: [B, T, H, W]
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_weighted

        T = focal_loss.shape[1]
        discounts = torch.pow(
            self.future_discount, torch.arange(T, device=inputs.device)
        )
        discounts = discounts.view(1, T, 1, 1)

        # Apply discount to the loss
        focal_loss = focal_loss * discounts

        # 6. Final reduction ignoring invalid indices
        if self.reduction == "mean":
            valid_mask = targets != self.ignore_index
            return (
                focal_loss[valid_mask].mean()
                if valid_mask.any()
                else torch.tensor(0.0, device=inputs.device)
            )
        elif self.reduction == "sum":
            valid_mask = targets != self.ignore_index
            return focal_loss[valid_mask].sum()
        else:
            return focal_loss