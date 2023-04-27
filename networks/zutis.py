from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from pycocotools.mask import encode
import clip
from networks.positional_embedding import PositionEmbeddingSine
from networks.clip_arch import build_model
from networks.transformer import TransformerDecoderLayer, TransformerDecoder
from utils.iou import compute_iou


class ZUTIS(nn.Module):
    def __init__(
            self,
            categories: List[str],
            segmentation_type: str = "semantic",
            clip_arch: str = "ViT-B/16",
            n_queries: int = 100,
            n_decoder_layers: int = 6,
            n_heads: int = 8,
            device: torch.device = torch.device("cuda:0"),
            encoder_type: str = "clip",
            frozen_bn: Optional[bool] = True,
            stop_gradient: Optional[bool] = True,
            decoder_image_n_dims: Optional[int] = None,
    ):
        """Define a encoder and decoder along with queries to be learned through the decoder."""
        super(ZUTIS, self).__init__()
        assert segmentation_type in ["semantic", "instance"], f"Invalid segmentation type: {segmentation_type}."

        # instantiate a CLIP visual encoder and extract text embeddings
        model, preprocess = clip.load(clip_arch.lstrip("dilated"), device=device)
        self.text_embeddings = model.encode_text(clip.tokenize(categories).to(device)).to(dtype=torch.float32).detach()
        self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=1, keepdim=True)
        self.text_embeddings.requires_grad_(False)
        self.category_to_text_embedding: Dict[str, torch.Tensor] = {
            category: text_embedding for category, text_embedding in zip(categories, self.text_embeddings)
        }
        self.n_dims_text: int = self.text_embeddings.shape[1]
        self.frozen_bn: bool = True if frozen_bn is None else frozen_bn
        self.stop_gradient: bool = True if stop_gradient is None else stop_gradient
        self.decoder_image_n_dims: int = decoder_image_n_dims

        if encoder_type == "clip" or encoder_type is None:
            encoder_type: str = "clip"
            
            # CLIP encoder
            clip_model = build_model(model.state_dict())
            print(
                f"# clip visual encoder ({clip_arch}) params:", sum(param.numel() for param in model.visual.parameters())
            )
            self.encoder = clip_model.visual.to(dtype=torch.float32)
            self.encoder.requires_grad_(True)
            self.encoder.train()

            self.ffn1 = MLP(
                input_dim=self.encoder.width,
                hidden_dim=256,
                output_dim=self.encoder.width,
                num_layers=3
            )

            self.ffn2 = MLP(
                input_dim=self.encoder.width,
                hidden_dim=256,
                output_dim=self.encoder.width,
                num_layers=3
            )

        elif encoder_type == "dino":
            # to compare zero-shot ability
            # DINO encoder
            import networks.vision_transformer as vits
            from utils.utils import load_model

            self.encoder = vits.__dict__["vit_base"](patch_size=16, num_classes=0)
            load_model(model=self.encoder, arch="vit_base", patch_size=16)
            self.encoder.requires_grad_(True)
            self.encoder.train()

            self.vision_to_text_ffn = MLP(
                input_dim=self.encoder.embed_dim,
                hidden_dim=256,
                output_dim=self.n_dims_text,
                num_layers=3
            )

            self.ffn1 = MLP(
                input_dim=self.n_dims_text,
                hidden_dim=256,
                output_dim=self.n_dims_text,
                num_layers=3
            )

            self.ffn2 = MLP(
                input_dim=self.n_dims_text,
                hidden_dim=256,
                output_dim=self.n_dims_text,
                num_layers=3
            )

        else:
            raise ValueError(encoder_type)
        print(f"{encoder_type} is loaded.")

        # positional encoding
        pos_dim_sq: int = self.encoder.width
        self.pe_layer = PositionEmbeddingSine(pos_dim_sq // 2, normalize=True)

        # instantiate a transformer decoder
        self.decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=self.encoder.width,
                nhead=n_heads,
                dropout=0.0
            ),
            num_layers=n_decoder_layers,
            norm=nn.LayerNorm(
                self.encoder.width
            ),
            return_intermediate=True
        )
        self.decoder.requires_grad_(True)
        self.decoder.train()

        # instantiate queries
        self.query_embed = nn.Embedding(
            n_queries,
            self.encoder.width
        ).weight  # initialized with gaussian(0, 1)

        self.n_queries: int = n_queries
        self.device: torch.device = device
        self.clip_arch: str = clip_arch
        self.encoder_type: str = encoder_type

    def forward_transformer_encoder(self, x: torch.Tensor):
        """
        :param x: b x c x h x w
        :return patch_tokens: b x depth x hw x n_dims
        """
        patch_tokens, h_feat, w_feat = self.encoder(x)
        return patch_tokens, h_feat, w_feat

    def forward_transformer_decoder(self, patch_tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Forward transformer decoder given patch tokens from the encoder's last layer.
        :param patch_tokens: b x hw x n_dims
        :param pos: b x n_dims x h x w
        :return queries: b (x n_layers) x n_queries x n_dims
        """
        b = patch_tokens.shape[0]
        patch_tokens = patch_tokens.permute(1, 0, 2)  # b x hw x n_dims -> hw x b x n_dims

        # queries: n_queries x n_dims -> n_queries x b x n_dims
        queries: torch.Tensor = self.query_embed.unsqueeze(1).repeat(1, b, 1)

        # pos: b x n_dims x h x w -> b x n_dims x hw -> hw x b x n_dims
        pos = pos.flatten(start_dim=2).permute(2, 0, 1)

        queries: torch.Tensor = self.decoder(
            tgt=torch.zeros_like(queries),
            memory=patch_tokens,
            pos=pos,
            query_pos=queries
        ).squeeze(dim=0)

        if len(queries.shape) == 3:
            queries: torch.Tensor = queries.permute(1, 0, 2)  # n_queries x b x n_dims -> b x n_queries x n_dims
        elif len(queries.shape) == 4:
            # n_layers x n_queries x b x n_dims -> b x n_layers x n_queries x n_dims
            queries: torch.Tensor = queries.permute(2, 0, 1, 3)
        return queries

    def get_mask_proposals(
            self,
            queries: torch.Tensor,
            patch_tokens: torch.Tensor,
            return_binary_masks: bool = True
    ):
        if len(queries.shape) == 3:
            mask_proposals: torch.Tensor = torch.einsum(
                "bqc,bhwc->bqhw", queries, patch_tokens
            )
            if return_binary_masks:
                b, n_queries, h, w = mask_proposals.shape
                binary_masks: torch.Tensor = torch.argmax(mask_proposals, dim=1)  # b x h x w
                one_hot_masks: torch.Tensor = torch.stack(
                    [binary_masks == query_index for query_index in range(n_queries)], dim=0
                ).permute(1, 0, 2, 3)
                return mask_proposals, one_hot_masks

        elif len(queries.shape) == 4:
            mask_proposals: torch.Tensor = torch.einsum(
                "bdqc,bhwc->bdqhw", queries, patch_tokens
            )

            if return_binary_masks:
                b, n_layers, n_queries, h, w = mask_proposals.shape
                binary_masks: torch.Tensor = torch.argmax(mask_proposals, dim=2)  # b x d x h x w
                one_hot_masks: torch.Tensor = torch.stack(
                    [binary_masks == query_index for query_index in range(n_queries)], dim=0
                ).permute(1, 2, 0, 3, 4)
                return mask_proposals, one_hot_masks
        else:
            raise ValueError(f"{len(queries.shape)} not in [3, 4]")
        return torch.sigmoid(mask_proposals)

    def non_maximum_suppression(
            self,
            image_id: int,
            binary_masks_per_image: np.ndarray,
            confidence_scores_per_image: np.ndarray,
            category_ids_per_image: np.ndarray,
            nms_type: str = "hard",
            nms_threshold: float = 0.3,
            sigma: float = 0.5,
            threshold: float = 0.001,
            label_id_to_category: Optional[Dict[int, str]] = None,
            new_label_id_to_old_label_id: Optional[Dict[int, int]] = None
    ):
        assert nms_type in ["hard", "linear", "gaussian"]
        if nms_type == "gaussian":
            assert isinstance(sigma, float), TypeError(sigma)
        elif nms_type == "linear":
            assert isinstance(nms_threshold, float), TypeError(nms_threshold)

        predictions_per_image: List[dict] = list()
        unique_category_ids = set(category_ids_per_image)
        for unique_category_id in unique_category_ids:
            if unique_category_id == 0:  # background category
                continue

            indices: np.ndarray = np.nonzero(category_ids_per_image == unique_category_id)  # M
            binary_masks_per_image_per_category: np.ndarray = binary_masks_per_image[indices]  # M x H x W
            confidence_scores_per_image_per_category: np.ndarray = confidence_scores_per_image[indices]  # M

            candidate_masks: np.ndarray = binary_masks_per_image_per_category
            candidate_scores: np.ndarray = confidence_scores_per_image_per_category

            selected_scores = list()
            selected_masks = list()
            while len(candidate_masks) > 0:
                sorted_indices: np.ndarray = np.argsort(candidate_scores)
                sorted_scores: np.ndarray = candidate_scores[sorted_indices]  # ascending
                sorted_mask: np.ndarray = candidate_masks[sorted_indices]

                max_score = sorted_scores[-1]
                max_mask = sorted_mask[-1]

                selected_scores.append(max_score)
                selected_masks.append(max_mask)

                candidate_scores: list = list()
                candidate_masks: list = list()
                for m, s in zip(sorted_mask[:-1], sorted_scores[:-1]):
                    iou: np.ndarray = compute_iou(pred_mask=m, gt_mask=max_mask)

                    if nms_type == "hard":
                        weight = 0 if iou > nms_threshold else 1
                    elif nms_type == "linear":
                        weight = (1 - iou) if iou > nms_threshold else 1
                    else:
                        weight = np.exp(-(iou * iou) / sigma)

                    s *= weight
                    if s > threshold:
                        candidate_scores.append(s)
                        candidate_masks.append(m)

                try:
                    candidate_scores: np.ndarray = np.array(candidate_scores)
                    candidate_masks: np.ndarray = np.stack(candidate_masks, axis=0)
                except ValueError:
                    # ValueError: need at least one array to stack -> no masks left -> break the loop
                    break

            for m, s in zip(selected_masks, selected_scores):
                if m.sum() == 0:
                    continue
                if new_label_id_to_old_label_id is not None:
                    label_id = new_label_id_to_old_label_id[unique_category_id.item()]
                else:
                    label_id = unique_category_id.item()

                prediction: dict = {
                    "category_id": label_id,
                    "segmentation": encode(np.asfortranarray(m)),
                    "score": s.item(),
                    "image_id": image_id,
                    "image_size": binary_masks_per_image[0].shape[-2:],
                    "bbox": masks_to_boxes(torch.from_numpy(m)[None])[0].numpy().tolist()
                }
                if label_id_to_category is not None:
                    prediction["pred_class"] = label_id_to_category[label_id]
                predictions_per_image.append(prediction)
        return predictions_per_image

    def image_to_text_space(
            self,
            patch_tokens: torch.Tensor,
            proj: torch.Tensor,
            channel_last: bool,
            layer_norm: bool = True
    ) -> torch.Tensor:
        if channel_last:
            if "RN" in self.clip_arch:
                b, h, w, c = patch_tokens.shape
                patch_tokens = patch_tokens.permute(0, 3, 1, 2)
                # patch_tokens: torch.Tensor = self.encoder.proj(patch_tokens).permute(0, 2, 3, 1)

                # # (h_feat x w_feat) x b x n_dims -> b x (h_feat x w_feat) x n_dims
                patch_tokens: torch.Tensor = self.encoder.proj(patch_tokens)[1:].permute(1, 0, 2)
                patch_tokens = patch_tokens.view(b, h, w, c)

            else:
                patch_tokens = torch.einsum("bhwn,nc->bhwc", patch_tokens, proj)
            if layer_norm:
                patch_tokens = F.layer_norm(patch_tokens, normalized_shape=(patch_tokens.shape[1:]))
            patch_tokens = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-7)
        else:
            if "RN" in self.clip_arch:
                patch_tokens = torch.einsum("bnhw,nc->bchw", patch_tokens, proj)
            else:
                patch_tokens: torch.Tensor = self.encoder.proj(patch_tokens)
            if layer_norm:
                patch_tokens = F.layer_norm(patch_tokens, normalized_shape=(patch_tokens.shape[1:]))
            patch_tokens = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + 1e-7)
        return patch_tokens

    def update_text_embeddings(self, categories):
        model, preprocess = clip.load(self.clip_arch.lstrip("dilated"), device=self.device)
        self.text_embeddings = model.encode_text(clip.tokenize(categories).to(self.device)).to(dtype=torch.float32).detach()
        self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=1, keepdim=True)
        self.text_embeddings.requires_grad_(False)
        print(f"text embeddings have been changed for {', '.join(categories)}")

    @torch.no_grad()
    def predict(
            self,
            dict_outputs: dict,
            mask_type: str,
            threshold: float = 0.5,  # threshold for binarising an instance mask
            image_ids: Optional[List[int]] = None,  # for COCO-style format
            size: Optional[Tuple[int, int]] = None,  # (H, W) format,
            label_id_to_category: Optional[Dict[int, str]] = None,  # for instance segmentation
            new_label_id_to_old_label_id: Optional[Dict[int, int]] = None,  # for coco labels
            temperature: float = 5,
            nms_type: str = "hard",
            return_logits: bool = False
    ):
        assert mask_type in ["semantic", "instance"]
        if mask_type == "semantic":
            # semantic prediction
            # patch_tokens: b x h x w x n_dims -> b x n_dims x H x W
            patch_tokens: torch.Tensor = dict_outputs["patch_tokens"].permute(0, 3, 1, 2)

            # semantic_prediction: n semantic categories x H x W
            semantic_predictions: torch.Tensor = torch.einsum(
                "nc,bchw->bnhw",
                self.text_embeddings,
                patch_tokens
            )
            if size is not None:
                semantic_predictions: torch.Tensor = F.interpolate(semantic_predictions, size=size, mode="bilinear")

            if return_logits:
                predictions: torch.Tensor = semantic_predictions
            else:
                predictions: np.ndarray = torch.argmax(semantic_predictions, dim=1).cpu().numpy()

        else:
            # instance prediction
            # mask_proposals (torch.float32): b x n_queries x h x w OR b x n_layers x n_queries x h x w
            mask_proposals: torch.Tensor = dict_outputs["mask_proposals"]

            if len(mask_proposals.shape) == 5:
                # consider mask proposals from the last decoder layer only
                # b x n_layers x n_queries x h x w - > b x n_queries x h x w
                mask_proposals = mask_proposals[:, -1, ...]

            # mask proposals should be composed of values between [0, 1]
            assert 0 <= torch.min(mask_proposals) <= 1
            assert 0 <= torch.max(mask_proposals) <= 1

            # decide a confidence score for each instance mask
            # binary_masks: b x n_queries x h x w
            binary_masks = mask_proposals > threshold

            # mask_sizes: b x n_queries
            mask_sizes = torch.sum(binary_masks, dim=(-2, -1))

            # confidence_scores: b x n_queries
            confidence_scores = torch.sum((mask_proposals * binary_masks), dim=(-2, -1)) / (mask_sizes + 1e-7)

            # decide a category of each instance mask based on average patch tokens within the mask region
            # patch_tokens: b x h x w x n_dims
            patch_tokens: torch.Tensor = dict_outputs["patch_tokens"]

            # patch_tokens: b x h x w x n_dims -> b x 1 x h x w x n_dims
            # average_patch_tokens: b x n_queries x n_dims
            average_patch_tokens: torch.Tensor = torch.sum(
                patch_tokens[:, None] * binary_masks[..., None], dim=(-3, -2)
            ) / (mask_sizes.unsqueeze(dim=-1) + 1e-7)

            # b x n_queries x n_categories
            semantic_predictions: torch.Tensor = torch.sigmoid(torch.einsum(
                "nc,bqc->bqn",
                self.text_embeddings,
                average_patch_tokens / (average_patch_tokens.norm(dim=-1, keepdim=True) + 1e-7)
            ) * temperature)

            # category_ids: b x n_queries
            category_ids: np.ndarray = torch.argmax(semantic_predictions, dim=-1).cpu().numpy()

            # a max category probability for each query to be multiplied for a confidence score
            max_category_probabilities = torch.max(semantic_predictions, dim=-1).values  # b x n_queries
            confidence_scores: np.ndarray = (confidence_scores * max_category_probabilities).cpu().numpy()

            if size is not None:
                # upsample instance masks now to run the inference efficiently
                mask_proposals: torch.Tensor = F.interpolate(mask_proposals, size=size, mode="bilinear")
                binary_masks: np.ndarray = (mask_proposals > threshold).cpu().numpy()
            else:
                binary_masks: np.ndarray = binary_masks.cpu().numpy()

            if image_ids is None:
                image_ids: List[int] = [0 for _ in range(len(binary_masks))]

            predictions: List[dict] = list()
            for batch_index, (binary_masks_per_image, confidence_scores_per_image, category_ids_per_image, image_id) in\
                    enumerate(zip(binary_masks, confidence_scores, category_ids, image_ids)
            ):
                if nms_type is None:
                    predictions_per_image: List[dict] = list()
                    for m, s, c in zip(binary_masks_per_image, confidence_scores_per_image, category_ids_per_image):
                        if m.sum() == 0 or c == 0:
                            continue

                        if new_label_id_to_old_label_id is not None:
                            label_id = new_label_id_to_old_label_id[c.item()]
                        else:
                            label_id = c.item()
                        prediction: dict = {
                            "category_id": label_id,
                            "segmentation": encode(np.asfortranarray(m)),
                            "score": s.item(),
                            "image_id": image_id,
                            "image_size": binary_masks_per_image[0].shape[-2:],
                            "bbox": masks_to_boxes(torch.from_numpy(m)[None])[0].numpy().tolist()
                        }
                        if label_id_to_category is not None:
                            prediction["pred_class"] = label_id_to_category[label_id]
                        predictions_per_image.append(prediction)

                else:
                    # nms
                    predictions_per_image = self.non_maximum_suppression(
                        image_id=image_id,
                        binary_masks_per_image=binary_masks_per_image,
                        confidence_scores_per_image=confidence_scores_per_image,
                        category_ids_per_image=category_ids_per_image,
                        label_id_to_category=label_id_to_category,
                        new_label_id_to_old_label_id=new_label_id_to_old_label_id,
                        nms_type=nms_type
                    )
                predictions.extend(predictions_per_image)
        return predictions

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: b x 3 x h x w
        """
        b, _, H, W = x.shape

        # patch_tokens: b x hw x n_dims
        patch_tokens, h_feat, w_feat = self.forward_transformer_encoder(x=x)
        _, hw, n_dims = patch_tokens.shape

        if self.encoder_type == "dino":
            # as the dimensionalities of visual features and text features are different in case of using DINO features,
            # we need to add a FFN as a bridge
            patch_tokens = self.vision_to_text_ffn(patch_tokens)
            _, hw, n_dims = patch_tokens.shape

        if "ViT" in self.clip_arch or "dilatedRN" not in self.clip_arch:
            # in case if it's a ViT-based or a not dilated ResNet based encoder, upsample patch tokens by 2.
            # patch_tokens: b x hw x n_dims -> b x h x w x n_dims -> b x 2h x 2w x n_dims
            patch_tokens = patch_tokens.view(b, h_feat, w_feat, n_dims).permute(0, 3, 1, 2)
            patch_tokens = F.interpolate(patch_tokens, scale_factor=2, mode="bilinear")

            # patch_tokens: b x 2h x 2w x n_dims -> b x 4hw x n_dims
            patch_tokens = patch_tokens.permute(0, 2, 3, 1).view(b, -1, n_dims)

            h_feat, w_feat = h_feat * 2, w_feat * 2

        # decoder_input: b x 4hw x n_dims
        if self.stop_gradient:
            decoder_input: torch.Tensor = self.ffn1(patch_tokens.detach())
        else:
            decoder_input: torch.Tensor = self.ffn1(patch_tokens)

        # transformer decoder forward
        # pos: b x n_dims x h_feat x w_feat, [-1, 1]
        pos = self.pe_layer(decoder_input.permute(0, 2, 1).view(b, -1, h_feat, w_feat))

        # queries: b x n_queries x n_dims OR b x n_layers x n_queries x n_dims
        queries: torch.Tensor = self.forward_transformer_decoder(
            patch_tokens=decoder_input,
            pos=pos
        )
        queries: torch.Tensor = self.ffn2(queries)
        queries = queries / queries.norm(dim=-1, keepdim=True)

        # patch_tokens: b x hw x n_dims -> b x h x w x n_dims
        patch_tokens = patch_tokens.view(b, h_feat, w_feat, n_dims)
        decoder_input = decoder_input.view(b, h_feat, w_feat, -1)

        # mask_proposals: b x n_queries x h x w OR b x n_layers x n_queries x h x w
        mask_proposals: torch.Tensor = self.get_mask_proposals(
            queries=queries,
            patch_tokens=decoder_input,
            return_binary_masks=False
        )

        patch_tokens: torch.Tensor = self.image_to_text_space(
            patch_tokens=patch_tokens, proj=self.encoder.proj, channel_last=True
        )

        return {"mask_proposals": mask_proposals, "patch_tokens": patch_tokens}


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
