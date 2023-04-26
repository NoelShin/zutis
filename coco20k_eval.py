import os
import requests
from typing import Dict, List, Optional
import yaml
import ujson as json
import numpy as np
import torch
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode
from utils.utils import get_network
from datasets.coco2017 import new_label_id_to_old_label_id
from tqdm import tqdm
from utils.iou import compute_iou
from detectron2.data import Metadata
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer


# adapted from https://leimao.github.io/blog/Inspecting-COCO-Dataset-Using-COCO-API/
def visualise(
        image_id: int, coco_api, fp: Optional[str] = None, draw_bbox: bool = False
):
    img_info = coco_gt.loadImgs([image_id])[0]
    img_file_name = img_info["file_name"]
    img_url = img_info["coco_url"]
    print(
        f"Image ID: {image_id}, File Name: {img_file_name}, Image URL: {img_url}"
    )

    # Get all the annotations for the specified image.
    ann_ids = coco_api.getAnnIds(imgIds=[image_id], iscrowd=None)
    anns = coco_api.loadAnns(ann_ids)
    print(f"Annotations for Image ID {image_id}:")

    # Use URL to load image.
    im = Image.open(requests.get(img_url, stream=True).raw)

    # Save image and its labeled version.
    plt.axis("off")
    plt.imshow(np.asarray(im))

    # Plot segmentation and bounding box.
    coco_api.showAnns(anns, draw_bbox=draw_bbox)

    if fp is not None:
        plt.savefig(fp, bbox_inches="tight", pad_inches=0)


def non_maximum_suppression(
    binary_masks_per_image: np.ndarray,
    confidence_scores_per_image: np.ndarray,
    category_ids_per_image: np.ndarray,
    nms_type: str = "hard",
    nms_threshold: float = 0.3,
    sigma: float = 0.5,
    threshold: float = 0.001,
    label_id_to_category: Optional[Dict[int, str]] = None
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

            label_id = new_label_id_to_old_label_id[unique_category_id.item()]
            prediction: dict = {
                "category_id": label_id,
                "segmentation": encode(np.asfortranarray(m)),
                "score": s.item(),
                "image_id": image_id,
                "image_size": binary_masks_per_image[0].shape[-2:],
                "bbox": masks_to_boxes(torch.from_numpy(m)[None])[0].numpy()
            }
            if label_id_to_category is not None:
                prediction["pred_class"] = label_id_to_category[label_id]
            predictions_per_image.append(prediction)
    return predictions_per_image


def convert_to_instances(predictions_per_image: List[dict]) -> Instances:
    pred_classes: List[int] = list()
    scores: List[float] = list()
    pred_masks: list = list()
    pred_boxes: list = list()  # np.ndarray, XYXY format
    for p in predictions_per_image:
        score = p["score"]
        if score > 0.5:
            pred_classes.append(p["category_id"])
            scores.append(score)
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


def visualise_instances(
        image: Image.Image,
        instances: Instances,
        label_id_to_category: Dict[int, str],
        label_id_to_rgb: Dict[int, np.ndarray],
        fp: Optional[str] = None,
        instance_mode: ColorMode = ColorMode.IMAGE
) -> VisImage:
    assert instance_mode in [ColorMode.IMAGE, ColorMode.SEGMENTATION], instance_mode

    np.random.seed(0)

    metadata: Metadata = Metadata()
    setattr(metadata, "thing_classes", label_id_to_category)
    setattr(metadata, "thing_colors", label_id_to_rgb)

    visualiser: Visualizer = Visualizer(
        img_rgb=np.array(image),
        metadata=metadata,
        instance_mode=instance_mode
    )

    visualisation: VisImage = visualiser.draw_instance_predictions(predictions=instances)
    if fp is not None:
        visualisation.save(fp)
        plt.close()
    return visualisation


if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    from utils.visualiser import Visualiser
    from utils.utils import get_label_id_to_category

    parser = ArgumentParser()
    parser.add_argument("--p_gt_annotations", type=str, default="/home/cs-shin1/islowp/configs/coco2017_acp_val.yaml")
    parser.add_argument("--eval_split", type=str, default="train2014_sel20k", choices=["train2014_sel20k", "val2017"])
    parser.add_argument("--p_config", type=str, required=True)
    parser.add_argument("--p_state_dict", type=str, required=True)
    parser.add_argument("--dir_dataset", type=str, default="/home/cs-shin1/datasets/coco")
    parser.add_argument("--dir_ckpt", type=str, default="/home/cs-shin1/zutis/ckpt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--visualise", type=bool, default=True)
    parser.add_argument("--nms_type", type=str, default="hard", choices=[None, "hard"])
    parser.add_argument("--suffix", type=str, default='')
    args = parser.parse_args()

    base_args = yaml.safe_load(open(f"{args.p_config}", 'r'))
    args: dict = vars(args)
    args.update(base_args)
    args: Namespace = Namespace(**args)

    device = torch.device("cuda:0")
    dir_ckpt = f"{args.dir_ckpt}/coco20k/{args.clip_arch.lower().replace('-', '_').replace('/', '_')}"
    os.makedirs(dir_ckpt, exist_ok=True)

    p_gt_annotations: str = f"{args.dir_dataset}/annotations/instances_{args.eval_split}.json"

    # instantiate a segmentation network
    network = get_network(network_name=args.clip_arch, categories=args.categories).to(device)
    network.load_state_dict(torch.load(args.p_state_dict))
    network.eval()
    network.requires_grad_(False)
    print(f"A model loaded from {args.p_state_dict}.")

    # instantiate a visualiser
    visualiser = Visualiser(label_id_to_category=get_label_id_to_category(dataset_name="coco2017"))

    coco_gt = COCO(p_gt_annotations)

    category_ids: List[int] = coco_gt.getCatIds()
    categories: List[dict] = coco_gt.loadCats(category_ids)
    image_ids: List[int] = coco_gt.getImgIds()

    predictions: List[dict] = list()
    list_image_ids = list()
    list_np_images = list()

    cnt = 0
    for num_image, image_id in enumerate(tqdm(image_ids)):
        cnt += 1
        img_info = coco_gt.loadImgs([image_id])[0]
        img_file_name = img_info["file_name"]
        img_url = img_info["coco_url"]

        # Get all the annotations for the specified image.
        ann_ids = coco_gt.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = coco_gt.loadAnns(ann_ids)

        p_image: str = f"{args.dir_dataset}/train2014/{img_file_name}"

        pil_image: Image.Image = Image.open(p_image).convert("RGB")
        list_np_images.append(np.array(pil_image))
        image: torch.Tensor = TF.normalize(TF.to_tensor(pil_image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        size = image.shape[-2:]

        dict_outputs = network(image[None].to(device))

        predictions_per_image: List[dict] = network.predict(
            dict_outputs=dict_outputs,
            mask_type="instance",
            size=size,
            image_ids=[image_id],
            new_label_id_to_old_label_id=new_label_id_to_old_label_id,
            nms_type=args.nms_type
        )
        predictions.extend(predictions_per_image)

        if args.visualise and num_image % 200 == 0:
            visualiser.visualise_instance_predictions(
                image=image.numpy(),
                predictions=predictions_per_image,
                confidence_threshold=0.85,
                fp=f"{dir_ckpt}/{image_id:06d}_detectron2_instance_vis.png"
            )

        list_image_ids.append(image_id)

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='segm')
    coco_eval.params.imgIds = list_image_ids

    # set parameters as desired
    n_max_detections = [1, 10, 100]
    coco_eval.params.useCats = True  # use_categories
    coco_eval.params.maxDets = n_max_detections

    coco_eval.evaluate()  # run per image evaluation
    coco_eval.accumulate()  # accumulate per image results
    coco_eval.summarize()  # display summary metrics of results

    eval_results: np.ndarray = coco_eval.stats

    dict_eval_results: Dict[str, float] = {
        "AP": eval_results[0],
        "AP_50": eval_results[1],
        "AP_75": eval_results[2],
        "AP_small": eval_results[3],
        "AP_medium": eval_results[4],
        "AP_large": eval_results[5],
        f"AR_{n_max_detections[0]}": eval_results[6],
        f"AR_{n_max_detections[1]}": eval_results[7],
        f"AR_{n_max_detections[2]}": eval_results[8],
        "AR_small": eval_results[9],
        "AR_medium": eval_results[10],
        "AR_large": eval_results[11]
    }

    if args.suffix != '':
        fp = f"{dir_ckpt}/coco20k_metrics_{args.clip_arch.lower().replace('/', '_').replace('-', '_')}_nms_{args.nms_type}_{args.suffix}.json"
    else:
        fp = f"{dir_ckpt}/coco20k_metrics_{args.clip_arch.lower().replace('/', '_').replace('-', '_')}_nms_{args.nms_type}.json"

    json.dump(dict_eval_results, open(fp, 'w'))
