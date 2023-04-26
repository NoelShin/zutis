from typing import Dict, List, Optional, Tuple, Union
from math import sqrt
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.data import Metadata
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer


class Visualiser:
    def __init__(self, label_id_to_category: Optional[Dict[int, str]] = None):
        self.label_id_to_category: Optional[Dict[int, str]] = label_id_to_category

    @staticmethod
    def numpy_to_pil(
            image: np.ndarray,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> Image.Image:
        assert isinstance(image, np.ndarray), TypeError(f"{type(image)} is not np.ndarray.")
        image = image * np.array(std)[:, None, None]
        image = image + np.array(mean)[:, None, None]
        image = image * 255.0
        image = np.clip(image, 0, 255)
        image: Image.Image = Image.fromarray(image.astype(np.uint8).transpose(1, 2, 0))
        return image

    @staticmethod
    def colourise_label(
            label: np.ndarray,
            palette: dict,
            ignore_index: int = 255
    ) -> np.ndarray:
        h, w = label.shape[-2:]
        coloured_label = np.zeros((h, w, 3), dtype=np.uint8)

        unique_label_ids = np.unique(label)
        for label_id in unique_label_ids:
            if label_id == ignore_index:
                coloured_label[label == label_id] = np.array([255, 255, 255], dtype=np.uint8)
            else:
                coloured_label[label == label_id] = palette[label_id]
        return coloured_label

    def visualise_semantic_predictions(
            self,
            image: Union[np.ndarray, Image.Image],  # 3 x H x W
            ground_truth: np.ndarray,  # H x W
            prediction: np.ndarray,  # H x W
            palette: Dict[int, Union[List[int], Tuple[int, int, int]]],
            ignore_index: Optional[int] = None,
            fp: Optional[str] = None
    ):
        if isinstance(image, np.ndarray):
            image: Image.Image = self.numpy_to_pil(image=image)
        else:
            assert isinstance(image, Image.Image), TypeError(type(image))

        # coloured_ground_truth, coloured_prediction: h x w x 3
        coloured_ground_truth: np.ndarray = self.colourise_label(
            label=ground_truth, palette=palette, ignore_index=ignore_index
        )
        coloured_prediction: np.ndarray = self.colourise_label(
            label=prediction, palette=palette, ignore_index=ignore_index
        )

        ncols = 3
        fig, ax = plt.subplots(nrows=1, ncols=ncols, squeeze=False, figsize=(ncols * 3, 3))
        for i in range(1):
            for j in range(ncols):
                if j == 0:
                    ax[i, j].imshow(image)
                    ax[i, j].set_xlabel("input")
                elif j == 1:
                    ax[i, j].imshow(coloured_ground_truth)
                    ax[i, j].set_xlabel("ground-truth")

                elif j == 2:
                    ax[i, j].imshow(coloured_prediction)
                    ax[i, j].set_xlabel("prediction")

                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        plt.tight_layout(pad=0.5)

        if fp is not None:
            plt.savefig(fp)
        plt.close()

    def __call__(
            self,
            image: np.ndarray,  # H x W x 3
            predictions: Union[np.ndarray, List[dict]],  # H x W or coco-style annotations
            ground_truths: Optional[Union[np.ndarray, List[dict]]] = None,  # H x W or coco-style annotations
            segmentation_type: str = "semantic",
            ignore_index: Optional[int] = 255,
            palette: Optional[Dict[int, Union[List[int], Tuple[int, int, int]]]] = None,
            # label_id_to_category: Optional[Dict[int, str]] = None,
            confidence_threshold: float = 0.5,
            fp: Optional[str] = None
    ):
        print("image", image.shape)
        assert segmentation_type in ["semantic", "instance"], ValueError(segmentation_type)
        if segmentation_type == "semantic":
            assert len(ground_truths.shape) == 2, ValueError(f"{len(ground_truths.shape)} != 2")
            assert ground_truths.shape == predictions.shape, ValueError(f"{ground_truths.shape} != {predictions.shape}")
            assert palette is not None, TypeError(palette)
            self.visualise_semantic_predictions(
                image=image,
                ground_truth=ground_truths,
                prediction=predictions,
                palette=palette,
                ignore_index=ignore_index,
                fp=fp
            )

        else:
            self.visualise_instance_predictions(
                image=image,
                predictions=predictions,
                confidence_threshold=confidence_threshold,
                fp=fp
            )

    @staticmethod
    def convert_to_instances(
            size: Union[List[int], Tuple[int, int]],  # (H, W)
            predictions: List[dict],
            confidence_threshold: float = 0.5
    ) -> Instances:
        pred_classes: List[int] = list()
        scores: List[float] = list()
        pred_masks: list = list()
        pred_boxes: list = list()  # np.ndarray, XYXY format
        for p in predictions:
            confidence = p["score"]
            if confidence > confidence_threshold:
                pred_classes.append(p["category_id"])
                scores.append(confidence)
                pred_masks.append(p["segmentation"])
                pred_boxes.append(p["bbox"])

        instances: Instances = Instances(
            image_size=size,
            pred_classes=np.array(pred_classes),
            scores=scores,
            pred_masks=pred_masks,
            pred_boxes=pred_boxes
        )
        return instances

    def visualise_instance_predictions(
        self,
        image: Union[np.ndarray, Image.Image],
        predictions: List[dict],  # coco-style annotations
        label_id_to_rgb: Dict[int, Union[List[int], Tuple[int, int, int]]] = None,  # effective when instance_mode = ColorMode.SEGMENTATION
        confidence_threshold: float = 0.75,
        fp: Optional[str] = None,
        instance_mode: ColorMode = ColorMode.IMAGE
    ):
        assert instance_mode in [ColorMode.IMAGE, ColorMode.SEGMENTATION]
        if isinstance(image, np.ndarray):
            image: Image.Image = self.numpy_to_pil(image=image)
        W, H = image.size
        instances: Instances = self.convert_to_instances(
            size=(H, W), predictions=predictions, confidence_threshold=confidence_threshold
        )

        # convert coco-style annotations to a detectron2 format
        np.random.seed(0)  # fix a seed for a reproducible palette

        metadata: Metadata = Metadata()
        setattr(metadata, "thing_classes", self.label_id_to_category)
        setattr(metadata, "thing_colors", label_id_to_rgb)

        detectron2_visualiser: Visualizer = Visualizer(
            img_rgb=np.array(image),
            metadata=metadata,
            instance_mode=instance_mode
        )

        visualisation: VisImage = detectron2_visualiser.draw_instance_predictions(predictions=instances)
        if fp is not None:
            visualisation.save(fp)
        plt.close()

    def visualise_mask_proposals(
            self,
            size: Union[List[int], Tuple[int, int]],  # (H, W)
            mask_proposals: torch.Tensor,
            fp: Optional[str] = None
    ):
        n_queries = len(mask_proposals)
        nrows = ncols = int(sqrt(n_queries))  # assume that n_queries is a square of an integer (e.g., 100)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols, nrows))
        cnt = 0
        for row_index in range(nrows):
            for col_index in range(ncols):
                ax[row_index, col_index].imshow(mask_proposals[cnt])
                ax[row_index, col_index].set_xticks([])
                ax[row_index, col_index].set_yticks([])
                cnt += 1
        plt.tight_layout(pad=0.1)
        if fp is not None:
            plt.savefig(fp)
        plt.close()

    def visualise_matched_mask_proposals(
            self,
            ground_truths,
            mask_proposals,
            instance_indices,
            query_indices,
            fp: Optional[str] = None
    ):
        nrows = 2  # first row: gt instance, second row: pred instance
        ncols = len(ground_truths)  # n_instances

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols, nrows))
        for row_index in range(nrows):
            for col_index in range(ncols):
                if row_index == 0:
                    ax[row_index, col_index].imshow(ground_truths[instance_indices[col_index]])
                    ax[row_index, col_index].set_xticks([])
                    ax[row_index, col_index].set_yticks([])
                else:
                    ax[row_index, col_index].imshow(mask_proposals[query_indices[col_index]])
                    ax[row_index, col_index].set_xticks([])
                    ax[row_index, col_index].set_yticks([])

        plt.tight_layout(pad=0.1)
        if fp is not None:
            plt.savefig(fp)
        plt.close()

