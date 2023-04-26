from typing import Optional, Union
import numpy as np
import torch


def compute_iou(
        pred_mask: Union[np.ndarray, torch.Tensor],
        gt_mask: Union[np.ndarray, torch.Tensor],
        threshold: Optional[float] = None,
        eps: float = 1e-7
) -> Union[np.ndarray, torch.Tensor]:
    """
    :param pred_mask: (H x W)
    :param gt_mask: (H x W), same shape as pred_mask
    :param threshold: a binarisation threshold
    :param eps: a small value for computational stability
    :return: (1)
    """
    assert pred_mask.shape == gt_mask.shape, f"{pred_mask.shape} != {gt_mask.shape}"
    assert len(pred_mask.shape) == 2, ValueError(f"{len(pred_mask.shape)} != 2")
    assert len(gt_mask.shape) == 2, ValueError(f"{len(gt_mask.shape)} != 2")

    mask = np.logical_and(0 <= gt_mask, gt_mask <= 1)
    valid_gt = gt_mask[mask]
    valid_pred = pred_mask[mask]

    if threshold is not None:
        valid_pred = valid_pred > threshold

    if isinstance(valid_pred, np.ndarray):
        intersection = np.logical_and(valid_pred, valid_gt).sum()
        union = np.logical_or(valid_pred, valid_gt).sum()
        iou = (intersection / (union + eps))
    else:
        intersection = torch.logical_and(valid_pred, valid_gt).sum()
        union = torch.logical_or(valid_pred, valid_gt).sum()
        iou = (intersection / (union + eps)).cpu()
    return iou