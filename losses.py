# https://github.com/maturk/dn-splatter/blob/main/dn_splatter/losses.py
"""Loss functions"""

import abc
from enum import Enum
from typing import Optional
from typing_extensions import Literal

import torch
from torch import Tensor, nn


class L1(nn.Module):
    """L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.abs(pred - gt).mean()
        else:
            return torch.abs(pred - gt)


class LogL1(nn.Module):
    """Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.log(1 + torch.abs(pred - gt)).mean()
        else:
            return torch.log(1 + torch.abs(pred - gt))


class EdgeAwareLogL1(nn.Module):
    """Gradient aware Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation
        self.logl1 = LogL1(implementation="per-pixel")

    def forward(self, pred: Tensor, gt: Tensor, rgb: Tensor, mask: Optional[Tensor]):
        logl1 = self.logl1(pred, gt)

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )
        lambda_x = torch.exp(-grad_img_x)
        lambda_y = torch.exp(-grad_img_y)

        loss_x = lambda_x * logl1[..., :, :-1, :]
        loss_y = lambda_y * logl1[..., :-1, :, :]

        if self.implementation == "per-pixel":
            if mask is not None:
                loss_x[~mask[..., :, :-1, :]] = 0
                loss_y[~mask[..., :-1, :, :]] = 0
            return loss_x[..., :-1, :, :] + loss_y[..., :, :-1, :]

        if mask is not None:
            assert mask.shape[:2] == pred.shape[:2]
            loss_x = loss_x[mask[..., :, :-1, :]]
            loss_y = loss_y[mask[..., :-1, :, :]]

        if self.implementation == "scalar":
            return loss_x.mean() + loss_y.mean()


class HuberL1(nn.Module):
    """L1+huber loss"""

    def __init__(
        self,
        tresh=0.2,
        implementation: Literal["scalar", "per-pixel"] = "scalar",
        **kwargs,
    ):
        super().__init__()
        self.tresh = tresh
        self.implementation = implementation

    def forward(self, pred, gt):
        mask = gt != 0
        l1 = torch.abs(pred[mask] - gt[mask])
        d = self.tresh * torch.max(l1)
        loss = torch.where(l1 < d, ((pred - gt) ** 2 + d**2) / (2 * d), l1)
        if self.implementation == "scalar":
            return loss.mean()
        else:
            return loss


class EdgeAwareTV(nn.Module):
    """Edge Aware Smooth Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, depth: Tensor, rgb: Tensor):
        """
        Args:
            depth: [batch, H, W, 1]
            rgb: [batch, H, W, 3]
        """
        grad_depth_x = torch.abs(depth[..., :, :-1, :] - depth[..., :, 1:, :])
        grad_depth_y = torch.abs(depth[..., :-1, :, :] - depth[..., 1:, :, :])

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


class TVLoss(nn.Module):
    """TV loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        Args:
            pred: [batch, H, W, 3]

        Returns:
            tv_loss: [batch]
        """
        h_diff = pred[..., :, :-1, :] - pred[..., :, 1:, :]
        w_diff = pred[..., :-1, :, :] - pred[..., 1:, :, :]
        return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))


class NormalLossType(Enum):
    """Enum for specifying depth loss"""

    L1 = "L1"
    Smooth = "Smooth"
    AdaptiveNormal = "AdaptiveNormal"


class NormalLoss(nn.Module):
    """Factory method class for various depth losses"""

    def __init__(self, normal_loss_type: NormalLossType, **kwargs):
        super().__init__()
        self.normal_loss_type = normal_loss_type
        self.kwargs = kwargs
        self.loss = self._get_loss_instance()

    @abc.abstractmethod
    def forward(self, *args) -> Tensor:
        return self.loss(*args)

    def _get_loss_instance(self) -> nn.Module:
        if self.normal_loss_type == NormalLossType.L1:
            return L1(**self.kwargs)
        elif self.normal_loss_type == NormalLossType.Smooth:
            return TVLoss(**self.kwargs)
        elif self.normal_loss_type == NormalLossType.AdaptiveNormal:
            return AdaptiveNormal(**self.kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {self.normal_loss_type}")


class AdaptiveNormal(nn.Module):
    """Adaptive loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation
        self.L1 = L1(implementation=self.implementation)

    def forward(self, pred, gt, step):
        if step < 15_000:
            return self.L1(pred, gt) * 0.5
        else:
            normal_diff = mean_angular_error(
                pred.permute(2, 0, 1).unsqueeze(0),
                gt.permute(2, 0, 1).unsqueeze(0),
            )
            normal_confidence = (normal_diff <= 0.1).squeeze(0)
            return self.L1(pred[normal_confidence, :], gt[normal_confidence, :])


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute the mean angular error between predicted and reference normals

    Args:
        predicted_normals: [B, C, H, W] tensor of predicted normals
        reference_normals : [B, C, H, W] tensor of gt normals

    Returns:
        mae: [B, H, W] mean angular error
    """
    dot_products = torch.sum(gt * pred, dim=1)  # over the C dimension
    # Clamp the dot product to ensure valid cosine values (to avoid nans)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    # Calculate the angle between the vectors (in radians)
    mae = torch.acos(dot_products)
    return mae