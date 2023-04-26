from typing import Dict, List, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


class Criterion:
    def __init__(
            self,
            text_embeddings: torch.Tensor,
            weight_ce_loss: float = 1.0,
            weight_mask_loss: float = 1.0,
            weight_dice_loss: float = 1.0,
            weight_bce_loss: float = 1.0,
            ignore_index: int = 255
    ):
        self.text_embeddings: torch.Tensor = text_embeddings  # n_categories x n_dims
        self.weight_ce_loss: float = weight_ce_loss
        self.weight_mask_loss: float = weight_mask_loss
        self.weight_dice_loss: float = weight_dice_loss
        self.weight_bce_loss: float = weight_bce_loss
        self.ignore_index: int = ignore_index

    # copy-pasted from https://github.com/NoelShin/selfmask/blob/master/criterion.py
    def dice_loss(self, dt_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dt_masks:
                    n_queries x (H * W), [0, 1]
            gt_masks:
                    n_instances x (H * W), {0, 1}
                    A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """

        numerator = 2 * torch.einsum("nc,mc->nm", dt_masks, gt_masks)
        denominator = dt_masks.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]
        dice_loss = 1 - (numerator + 1) / (denominator + 1)  # n_queries x n_instances
        return dice_loss

    @staticmethod
    def binary_cross_entropy_loss(dt_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dt_masks:
                    n_queries x (H * W), [0, 1]
            gt_masks:
                    n_instances x (H * W), {0, 1}
                    A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        n_queries = dt_masks.shape[0]
        n_instances = gt_masks.shape[0]

        # n_queries x n_instances
        return F.binary_cross_entropy(
            dt_masks[:, None].repeat(1, n_instances, 1), gt_masks[None].repeat(n_queries, 1, 1), reduction="none"
        ).mean(dim=-1)

    def __call__(
            self,
            batch_mask_proposals: torch.Tensor,  # b (x n_layers) x n_queries x H x W
            batch_ground_truth_instance_masks: List[torch.Tensor],  # b x n_instances (variable) x H x W, {0, 1}
            batch_category_ids: List[List[int]],  # b x n_instances (variable), [0, n_categories - 1]
            batch_patch_tokens: torch.Tensor,  # b x h x w x text_dims
            batch_ground_truth_semantic_masks,  # b x H x W
    ) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
        """
        Note: compute all the possible losses between the predictions and labels and pick the least losses as many as
        batch_size
        """
        assert 0 <= batch_mask_proposals.min() <= 1, f"unexpected value: {batch_mask_proposals.min()}"
        assert 0 <= batch_mask_proposals.max() <= 1, f"unexpected value: {batch_mask_proposals.max()}"
        batch_size = len(batch_mask_proposals)

        # ===== cross-entropy loss =====
        # patch_tokens: b x H x W x n_dims -> b x n_dims x H x W
        batch_patch_tokens = batch_patch_tokens.permute(0, 3, 1, 2)
        batch_patch_tokens = F.interpolate(
            batch_patch_tokens, size=batch_ground_truth_instance_masks[0].shape[-2:], mode="bilinear"
        )

        semantic_predictions = torch.einsum(
            "nc,bchw->bnhw", self.text_embeddings, batch_patch_tokens
        )

        ce_loss = F.cross_entropy(
            semantic_predictions,
            batch_ground_truth_semantic_masks.to(batch_patch_tokens.device),
            ignore_index=self.ignore_index
        )
        # =============================

        # ===== mask loss =====
        mask_loss: torch.Tensor = torch.tensor(0., device=batch_mask_proposals.device, requires_grad=True)
        # iterate over batches
        for batch_index, (mask_proposals, ground_truth_instance_masks) in enumerate(
                zip(batch_mask_proposals, batch_ground_truth_instance_masks)
        ):
            assert len(ground_truth_instance_masks.shape) == 3, \
                f"Invalid ground truth instance masks shape: {len(ground_truth_instance_masks.shape)} != 3"

            # ground_truth_instance_masks: n_instances x H x W
            ground_truth_instance_masks: torch.Tensor = ground_truth_instance_masks.to(
                device=mask_proposals.device, dtype=torch.float32
            )
            H, W = ground_truth_instance_masks.shape[-2:]

            # ground_truth_instance_masks: n_instances x H x W -> n_instances x HW
            ground_truth_instance_masks = ground_truth_instance_masks.flatten(start_dim=-2)

            if ground_truth_instance_masks.sum() == 0:
                # in case there is no object in a batch of ground-truth masks for an image.
                continue

            if len(mask_proposals.shape) == 3:
                # mask_proposals: n_queries x h x w -> 1 x n_queries x hw
                mask_proposals = mask_proposals.unsqueeze(dim=0)

            # mask_proposals: n_layers x n_queries x h x w -> n_layers x n_queries x H x W -> n_layers x n_queries x HW
            mask_proposals = F.interpolate(mask_proposals, size=(H, W), mode="bilinear")
            mask_proposals = mask_proposals.flatten(start_dim=-2)

            # iterate over the transformer decoder layers
            mask_loss_per_image = 0.

            for layer_index, mask_proposals_per_layer in enumerate(mask_proposals):
                # mask_proposals_per_layer: n_queries x hw

                # dice_loss: n_queries x n_instances -> n_instances x n_queries
                dice_loss: torch.Tensor = self.dice_loss(
                    dt_masks=mask_proposals_per_layer, gt_masks=ground_truth_instance_masks
                ).permute(1, 0)

                # bce_loss: n_queries x n_instances -> n_instances x n_queries
                bce_loss: torch.Tensor = self.binary_cross_entropy_loss(
                    dt_masks=mask_proposals_per_layer, gt_masks=ground_truth_instance_masks
                ).permute(1, 0)

                cost_matrix: torch.Tensor = self.weight_dice_loss * dice_loss + self.weight_bce_loss * bce_loss

                instance_indices, query_indices = linear_sum_assignment(cost_matrix=cost_matrix.detach().cpu().numpy())
                mask_loss_per_layer = 0.
                for instance_index, query_index in zip(instance_indices, query_indices):
                    mask_loss_per_layer = mask_loss_per_layer + cost_matrix[instance_index, query_index]
                mask_loss_per_image += mask_loss_per_layer
            mask_loss = mask_loss + mask_loss_per_image
        mask_loss = mask_loss / batch_size
        # ===================

        loss = self.weight_mask_loss * mask_loss + self.weight_ce_loss * ce_loss
        return {
            "ce_loss": ce_loss.detach().cpu().item(),
            "mask_loss": mask_loss.detach().cpu().item(),
            "loss": loss,
            "instance_indices": instance_indices,  # for visualisation purpose
            "query_indices": query_indices  # for visualisation purpose
        }
