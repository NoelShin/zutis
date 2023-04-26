from typing import Dict, List
from math import sqrt, log
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from networks.selfmask.transformer_decoder import TransformerDecoderLayer, TransformerDecoder
from utils.bilateral_solver import bilateral_solver_output
from utils.utils import convert_tensor_to_pil_image


class SelfMask(nn.Module):
    def __init__(
            self,
            n_queries: int = 20,
            patch_size: int = 8,
            n_decoder_layers: int = 6,
            normalize_before: bool = False,
            return_intermediate: bool = False,
            scale_factor: int = 2,
            use_binary_classifier: bool = True
    ):
        """Define a encoder and decoder along with queries to be learned through the decoder."""
        super(SelfMask, self).__init__()
        import networks.selfmask.vision_transformer as vits
        self.encoder = vits.__dict__["deit_small"](patch_size=patch_size, num_classes=0)
        n_dims: int = self.encoder.n_embs
        n_heads: int = self.encoder.n_heads
        mlp_ratio: int = self.encoder.mlp_ratio

        decoder_layer = TransformerDecoderLayer(
            n_dims, n_heads, n_dims * mlp_ratio, 0., activation="relu", normalize_before=normalize_before
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            n_decoder_layers,
            norm=nn.LayerNorm(n_dims),
            return_intermediate=return_intermediate
        )

        self.query_embed = nn.Embedding(n_queries, n_dims).weight  # initialized with gaussian(0, 1)

        self.ffn = MLP(n_dims, n_dims, 1, num_layers=3)

        self.arch = "vit_small"
        self.use_binary_classifier = use_binary_classifier
        self.scale_factor = scale_factor

    # copy-pasted from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    @staticmethod
    def positional_encoding_2d(n_dims: int, height: int, width: int):
        """
        :param n_dims: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if n_dims % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(n_dims))
        pe = torch.zeros(n_dims, height, width)
        # Each dimension use half of d_model
        d_model = int(n_dims / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe

    def forward_encoder(self, x: torch.Tensor):
        """
        :param x: b x c x h x w
        :return patch_tokens: b x depth x hw x n_dims
        """
        if self.arch == "vit_small":
            encoder_outputs: Dict[str, torch.Tensor] = self.encoder(x)  # [:, 1:, :]
            all_patch_tokens: List[torch.Tensor] = list()
            for layer_name in [f"layer{num_layer}" for num_layer in range(1, self.encoder.depth + 1)]:
                patch_tokens: torch.Tensor = encoder_outputs[layer_name][:, 1:, :]  # b x hw x n_dims
                all_patch_tokens.append(patch_tokens)

            all_patch_tokens: torch.Tensor = torch.stack(all_patch_tokens, dim=0)  # depth x b x hw x n_dims
            all_patch_tokens = all_patch_tokens.permute(1, 0, 3, 2)  # b x depth x n_dims x hw
            return all_patch_tokens
        else:
            encoder_outputs = self.linear_layer(self.encoder(x)[-1])  # b x n_dims x h x w
            return encoder_outputs

    def forward_transformer_decoder(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Forward transformer decoder given patch tokens from the encoder's last layer.
        :param patch_tokens: b x n_dims x hw -> hw x b x n_dims
        :return queries: n_queries x b x n_dims -> b x n_queries x n_dims or b x n_layers x n_queries x n_dims
        """
        b = patch_tokens.shape[0]
        patch_tokens = patch_tokens.permute(2, 0, 1)  # b x n_dims x hw -> hw x b x n_dims

        # n_queries x n_dims -> n_queries x b x n_dims
        queries: torch.Tensor = self.query_embed.unsqueeze(1).repeat(1, b, 1)
        queries: torch.Tensor = self.decoder.forward(
            tgt=torch.zeros_like(queries),
            memory=patch_tokens,
            query_pos=queries
        ).squeeze(dim=0)

        if len(queries.shape) == 3:
            queries: torch.Tensor = queries.permute(1, 0, 2)  # n_queries x b x n_dims -> b x n_queries x n_dims
        elif len(queries.shape) == 4:
            # n_layers x n_queries x b x n_dims -> b x n_layers x n_queries x n_dims
            queries: torch.Tensor = queries.permute(2, 0, 1, 3)
        return queries

    def forward_pixel_decoder(self, patch_tokens: torch.Tensor, input_size=None):
        """ Upsample patch tokens by self.scale_factor and produce mask predictions
        :param patch_tokens: b (x depth) x n_dims x hw -> b (x depth) x n_dims x h x w
        :param queries: b x n_queries x n_dims
        :return mask_predictions: b x n_queries x h x w
        """

        if input_size is None:
            # assume square shape features
            hw = patch_tokens.shape[-1]
            h = w = int(sqrt(hw))
        else:
            # arbitrary shape features
            h, w = input_size
        patch_tokens = patch_tokens.view(*patch_tokens.shape[:-1], h, w)

        assert len(patch_tokens.shape) == 4
        patch_tokens = F.interpolate(patch_tokens, scale_factor=self.scale_factor, mode="bilinear")
        return patch_tokens

    def forward(self, x, encoder_only=False, inference: bool = False, bilateral_solver: bool = False):
        """
        x: b x c x h x w
        patch_tokens: b x n_patches x n_dims -> n_patches x b x n_dims
        query_emb: n_queries x n_dims -> n_queries x b x n_dims
        """
        b, _, H, W = x.shape
        dict_outputs: dict = dict()

        # b x depth x n_dims x hw (vit) or b x n_dims x h x w (resnet50)
        features: torch.Tensor = self.forward_encoder(x)

        if self.arch == "vit_small":
            # extract the last layer for decoder input
            last_layer_features: torch.Tensor = features[:, -1, ...]  # b x n_dims x hw
        else:
            # transform the shape of the features to the one compatible with transformer decoder
            b, n_dims, h, w = features.shape
            last_layer_features: torch.Tensor = features.view(b, n_dims, h * w)  # b x n_dims x hw

        if encoder_only:
            _h, _w = self.encoder.make_input_divisible(x).shape[-2:]
            _h, _w = _h // self.encoder.patch_size, _w // self.encoder.patch_size

            b, n_dims, hw = last_layer_features.shape
            dict_outputs.update({"patch_tokens": last_layer_features.view(b, _h, _w, n_dims)})
            return dict_outputs

        # transformer decoder forward
        queries: torch.Tensor = self.forward_transformer_decoder(
            last_layer_features,
        )  # b x n_queries x n_dims or b x n_layers x n_queries x n_dims

        # pixel decoder forward (upsampling the patch tokens by self.scale_factor)
        if self.arch == "vit_small":
            _h, _w = self.encoder.make_input_divisible(x).shape[-2:]
            _h, _w = _h // self.encoder.patch_size, _w // self.encoder.patch_size
        else:
            _h, _w = h, w
        features: torch.Tensor = self.forward_pixel_decoder(
            patch_tokens=last_layer_features,
            input_size=(_h, _w)
        )  # b x n_dims x h x w

        if len(queries.shape) == 3:
            # queries: b x n_queries x n_dims

            # mask_pred: b x n_queries x h x w -> b x 1 x n_queries x h x w
            mask_pred = torch.sigmoid(torch.einsum("bqn,bnhw->bqhw", queries, features)[:, None])
            objectness = self.ffn(queries)[:, None]  # b x n_queries x 1 -> b x 1 x n_queries x 1

        else:
            # queries: b x n_layers x n_queries x n_dims
            # mask_pred: b x n_layers x n_queries x h x w
            mask_pred = torch.sigmoid(torch.einsum("bdqn,bnhw->bdqhw", self.ffn(queries), features))

            # queries: b x n_layers x n_queries x n_dims -> n_layers x b x n_queries x n_dims
            queries = queries.permute(1, 0, 2, 3)

            objectness: List[torch.Tensor] = list()
            for n_layer, queries_per_layer in enumerate(queries):  # queries_per_layer: b x n_queries x n_dims
                objectness_per_layer = self.ffn(queries_per_layer)  # b x n_queries x 1
                objectness.append(objectness_per_layer)

            # n_layers x b x n_queries x 1 -> # b x n_layers x n_queries x 1
            objectness: torch.Tensor = torch.stack(objectness).permute(1, 0, 2, 3)

        if inference:
            # resize the prediction to input and apply bilateral solver if necessary

            mask_pred: torch.Tensor = mask_pred[:, 0]  # b x 1 x n_queries x H x W -> b x n_queries x H x W
            mask_pred = F.interpolate(mask_pred, scale_factor=4, mode="bilinear", align_corners=False)
            mask_pred = mask_pred[..., :H, :W]  # b x n_queries x H x W

            # objectness: b x 1 x n_queries x 1 -> b x n_queries
            objectness = objectness[:, 0, :, 0]

            dts: List[torch.Tensor] = list()
            dts_bi: List[torch.Tensor] = list()
            for num_image, (_mask_pred, _objectness) in enumerate(zip(mask_pred, objectness)):
                # _mask_pred: n_queries x H x W
                # _objectness: n_queries
                index_max_objectness = torch.argmax(_objectness)
                dt: torch.Tensor = _mask_pred[index_max_objectness]
                dt: torch.Tensor = (dt > 0.5).cpu().to(torch.uint8)
                dts.append(dt)
                # dt: np.ndarray = (dt > 0.5).cpu().numpy()
                # dts.append(dt.astype(np.uint8) * 255)

                if bilateral_solver:
                    _x: torch.Tensor = x[num_image]  # 3 x H x W
                    _pil_image: Image.Image = convert_tensor_to_pil_image(_x)

                    dt_bi, _ = bilateral_solver_output(img=_pil_image, target=dt.numpy())  # float64
                    dt_bi: np.ndarray = (dt_bi > 0.5)
                    dt_bi: torch.Tensor = torch.from_numpy(np.clip(dt_bi, 0, 1).astype(np.uint8)).to(torch.uint8)
                    dts_bi.append(dt_bi)

            dict_outputs["dts"] = dts
            if bilateral_solver:
                dict_outputs["dts_bi"] = dts_bi

        else:
            dict_outputs.update({
                "objectness": torch.sigmoid(objectness),
                "mask_pred": mask_pred
            })

        return dict_outputs


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


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, n_groups=32, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.ReLU()
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(self.block(x), scale_factor=self.scale_factor, mode="bilinear")
