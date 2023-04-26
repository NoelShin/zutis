import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import ujson as json
import numpy as np
import torch
from torch.nn import BatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.running_score import RunningScore
from utils.visualiser import Visualiser
from datasets.coco2017 import new_label_id_to_old_label_id


class Trainer:
    def __init__(
            self,
            network: nn.Module,
            device: torch.device = torch.device("cuda:0"),
            dir_ckpt: Optional[str] = None,
            palette: Optional[Dict[int, Tuple[int, int, int]]] = None,
            visualiser: Optional[Visualiser] = None,
            debug: bool = False
    ):
        self.network: nn.Module = network
        self.device: torch.device = device
        self.dir_ckpt: Optional[str] = dir_ckpt
        self.palette: Optional[Dict[int, Tuple[int, int, int]]] = palette
        self.visualiser: Visualiser = Visualiser() if visualiser is None else visualiser
        self.debug: bool = debug
        self.best_miou: float = -1.

    def visualise(
            self,
            fp: str,
            img: np.ndarray,
            gt: np.ndarray,
            dt: np.ndarray,
            palette: dict,
            dt_crf: Optional[np.ndarray] = None,
            ignore_index: int = 255
    ):
        def colourise_label(label: np.ndarray, palette: dict, ignore_index: int = 255) -> np.ndarray:
            h, w = label.shape[-2:]
            coloured_label = np.zeros((h, w, 3), dtype=np.uint8)

            unique_label_ids = np.unique(label)
            for label_id in unique_label_ids:
                if label_id == ignore_index:
                    coloured_label[label == label_id] = np.array([255, 255, 255], dtype=np.uint8)
                else:
                    coloured_label[label == label_id] = palette[label_id]
            return coloured_label

        img = img * np.array([0.229, 0.224, 0.225])[:, None, None]
        img = img + np.array([0.485, 0.456, 0.406])[:, None, None]
        img = img * 255.0
        img = np.clip(img, 0, 255)
        img: Image.Image = Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0))

        coloured_gt: np.ndarray = colourise_label(label=gt, palette=palette, ignore_index=ignore_index)  # h x w x 3
        coloured_dt: np.ndarray = colourise_label(label=dt, palette=palette, ignore_index=ignore_index)  # h x w x 3
        if dt_crf is not None:
            coloured_dt_crf: np.ndarray = colourise_label(label=dt_crf, palette=palette, ignore_index=ignore_index)  # h x w x 3

        ncols = 4 if dt_crf is not None else 3
        fig, ax = plt.subplots(nrows=1, ncols=ncols, squeeze=False, figsize=(ncols * 3, 3))
        for i in range(1):
            for j in range(ncols):
                if j == 0:
                    ax[i, j].imshow(img)
                    ax[i, j].set_xlabel("input")
                elif j == 1:
                    ax[i, j].imshow(coloured_gt)
                    ax[i, j].set_xlabel("ground-truth")

                elif j == 2:
                    ax[i, j].imshow(coloured_dt)
                    ax[i, j].set_xlabel("output")

                elif j == 3 and dt_crf is not None:
                    ax[i, j].imshow(coloured_dt_crf)
                    ax[i, j].set_xlabel("output (crf)")

                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        plt.tight_layout(pad=0.5)
        plt.savefig(fp)
        plt.close()

    def set_train_mode(self):
        self.network.train()

        if "RN" in self.network.clip_arch and self.network.frozen_bn:
            for m in self.network.encoder.modules():
                if isinstance(m, BatchNorm2d):
                    m.requires_grad_(False)
                    m.eval()

    def fit(
            self,
            dataloader: DataLoader,
            criterion: callable,
            optimiser: torch.optim.Optimizer,
            n_iters: int,
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            metric_meter: Optional = None,
            iter_eval: Optional[int] = None,
            iter_log: Optional[int] = None,
            val_dataloader: Optional[DataLoader] = None,
    ):
        self.set_train_mode()

        loss_meter = AverageMeter()

        iter_dataloader, pbar = iter(dataloader), tqdm(range(1, n_iters + 1))
        for num_iter in pbar:
            try:
                dict_data = next(iter_dataloader)
            except StopIteration:
                del iter_dataloader
                iter_dataloader = iter(dataloader)
                dict_data = next(iter_dataloader)

            # image: b x 3 x H x W, torch.float32
            # instance_mask: b x n_instances (variable) x H x W, torch.in64
            image, instance_mask = dict_data["image"], dict_data["instance_mask"]
            H, W = instance_mask[0].shape[-2:]

            # forward
            dict_outputs: Dict[str, torch.Tensor] = self.network(image.to(self.device))

            # backward
            dict_losses = criterion(
                batch_mask_proposals=dict_outputs["mask_proposals"],  # b (x n_layers) x n_queries x h x w
                batch_ground_truth_instance_masks=dict_data["instance_mask"],
                batch_category_ids=dict_data["category_ids"],
                batch_patch_tokens=dict_outputs["patch_tokens"],
                batch_ground_truth_semantic_masks=dict_data["semantic_mask"],
            )

            loss = dict_losses["loss"]
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loss_meter.update(loss.detach().cpu().item(), 1)

            if lr_scheduler is not None:
                lr_scheduler.step()

            with torch.no_grad():
                # compute metrics
                # patch_tokens: b x h x w x n_dims -> b x n_dims x h x w -> b x n_dims x H x W
                patch_tokens = dict_outputs["patch_tokens"].permute(0, 3, 1, 2)

                # semantic_prediction: n semantic categories x H x W
                semantic_prediction = torch.einsum(
                    "nc,bchw->bnhw",
                    self.network.text_embeddings,
                    patch_tokens
                )

                # upsample
                semantic_prediction = F.interpolate(
                    semantic_prediction, size=(H, W), mode="bilinear"
                )

                semantic_prediction: np.ndarray = torch.argmax(semantic_prediction, dim=1).cpu().numpy()
                semantic_mask: np.ndarray = dict_data["semantic_mask"].cpu().numpy()

                # compute metrics
                metric_meter.update(semantic_mask, semantic_prediction)
                scores: Tuple[Dict[str, float], Dict[str, float]] = metric_meter.get_scores()
                miou, pixel_acc = scores[0]["Mean IoU"], scores[0]["Pixel Acc"]

                pbar.set_description(
                    f"({num_iter}/{n_iters}) | "
                    f"Loss: {loss_meter.avg:.3f} | "
                    f"mask loss: {dict_losses['mask_loss']:.3f} | "
                    f"ce loss: {dict_losses['ce_loss']:.3f} | "
                    f"mIoU: {miou:.3f} | "
                    f"pixel acc.: {pixel_acc:.3f}"
                )

                # save training metrics
                if self.debug or isinstance(iter_log, int) and num_iter % iter_log == 0 and self.dir_ckpt is not None:
                    results: dict = {"num_iter": num_iter, "timestamp": str(datetime.now())}
                    results.update(scores[0])
                    results.update(scores[1])

                    if num_iter == iter_log:
                        json.dump(results, open(f"{self.dir_ckpt}/training_metrics.json", 'w'))
                    else:
                        with open(f"{self.dir_ckpt}/training_metrics.json", 'a') as f:
                            f.write('\n')
                            json.dump(results, f)
                            f.close()

                    # visualise semantic predictions
                    if self.palette is not None:
                        os.makedirs(f"{self.dir_ckpt}/train_images", exist_ok=True)
                        self.visualiser.visualise_semantic_predictions(
                            image=image[-1].numpy(),
                            ground_truth=semantic_mask[-1],
                            prediction=semantic_prediction[-1],
                            palette=self.palette,
                            fp=f"{self.dir_ckpt}/train_images/{num_iter:05d}_semantic.png"
                        )

                    mask_proposals = dict_outputs["mask_proposals"]
                    if len(mask_proposals.shape) == 4:
                        mask_proposals = mask_proposals[-1]  # b x n_queries x h x w -> n_queries x h x w
                    elif len(mask_proposals.shape) == 5:
                        mask_proposals = mask_proposals[-1, -1]  # b x n_layers x n_queries x h x w -> n_queries x h x w
                    else:
                        raise ValueError(mask_proposals.shape)

                    # mask_proposals: n_queries x h x w -> n_queries x H x W
                    mask_proposals = F.interpolate(mask_proposals[None], size=(H, W), mode="bilinear")[0].cpu().numpy()

                    # visualise mask proposals instance predictions
                    self.visualiser.visualise_mask_proposals(
                        size=image.shape[-2:],
                        mask_proposals=mask_proposals,
                        fp=f"{self.dir_ckpt}/train_images/{num_iter:05d}_mask_proposals.png"
                    )

                    # visualise matched instance predictions
                    self.visualiser.visualise_matched_mask_proposals(
                        ground_truths=instance_mask[-1],
                        mask_proposals=mask_proposals,
                        instance_indices=dict_losses["instance_indices"],
                        query_indices=dict_losses["query_indices"],
                        fp=f"{self.dir_ckpt}/train_images/{num_iter:05d}_gt_instance.png",
                    )

            # evaluate the model
            if (self.debug or num_iter % iter_eval == 0) and val_dataloader is not None \
                    and val_dataloader.dataset.name != "imagenet-s919":
                self.evaluate(dataloader=val_dataloader, num_iter=num_iter, iter_eval=iter_eval)
                torch.save(self.network.state_dict(), f"{self.dir_ckpt}/latest_model.pt")
                self.set_train_mode()

            if self.debug:
                break

        torch.save(self.network.state_dict(), f"{self.dir_ckpt}/final_model.pt")
        print(f"The final model is saved at {self.dir_ckpt}/final_model.pt.")

    def compute_coco_metrics(
            self,
            p_annotations: str,
            instance_predictions,
            use_categories: bool = True,
            n_max_detections: Tuple[int, ...] = (1, 10, 100),
    ) -> Dict[str, float]:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO(p_annotations)
        coco_dt = coco_gt.loadRes(instance_predictions)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='segm')

        # set parameters as desired
        coco_eval.params.useCats = use_categories
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
        return dict_eval_results

    @torch.no_grad()
    def evaluate(
            self,
            dataloader: DataLoader,
            num_iter: Optional[int] = None,
            iter_eval: Optional[int] = None,
            p_state_dict: Optional[str] = None
    ):
        if p_state_dict is not None:
            state_dict = torch.load(p_state_dict)
            self.network.load_state_dict(state_dict=state_dict, strict=True)
            print(f"Pre-trained parameters are loaded from {p_state_dict}.")
            num_iter: int = 0

        self.network.eval()
        dataset_name: str = dataloader.dataset.name
        n_categories: int = dataloader.dataset.n_categories

        list_instance_predictions: list = list()

        metric_meter = RunningScore(n_classes=n_categories)
        iter_dataloader, pbar = iter(dataloader), tqdm(range(len(dataloader)))
        for i in pbar:
            dict_data = next(iter_dataloader)

            image: torch.Tensor = dict_data["image"]
            semantic_masks: np.ndarray = dict_data["semantic_mask"].cpu().numpy()

            if "imagenet-s" in dataset_name:
                H, W = dict_data["original_size"]
            else:
                H, W = image.shape[-2:]

            # forward
            dict_outputs: Dict[str, torch.Tensor] = self.network(image.to(self.device))

            # semantic predictions: b x H x W
            semantic_predictions: np.ndarray = self.network.predict(
                dict_outputs=dict_outputs, mask_type="semantic", size=(H, W)
            )

            if dataset_name in ["coco2017", "voc2012"]:
                # instance predictions
                instance_predictions: List[dict] = self.network.predict(
                    dict_outputs=dict_outputs,
                    mask_type="instance",
                    size=(H, W),
                    image_ids=dict_data.get("image_id", None),
                    new_label_id_to_old_label_id=new_label_id_to_old_label_id if dataset_name == "coco2017" else None,
                    nms_type=None if ("no_sg" in self.dir_ckpt and dataset_name == "voc2012") else "hard"
                )
                list_instance_predictions.extend(instance_predictions)

            metric_meter.update(semantic_masks, semantic_predictions)
            scores: Tuple[Dict[str, float], Dict[str, float]] = metric_meter.get_scores()
            miou, pixel_acc = scores[0]["Mean IoU"], scores[0]["Pixel Acc"]

            pbar.set_description(
                f"({num_iter}) | "
                f"mIoU: {miou:.3f} | "
                f"pixel acc.: {pixel_acc:.3f}"
            )

            if self.debug or self.palette is not None and i % 100 == 0:
                os.makedirs(f"{self.dir_ckpt}/eval_images/{num_iter:05d}", exist_ok=True)
                self.visualiser.visualise_semantic_predictions(
                    image=image[0].numpy(),
                    ground_truth=semantic_masks[0],
                    prediction=semantic_predictions[0],
                    palette=self.palette,
                    fp=f"{self.dir_ckpt}/eval_images/{num_iter:05d}/{i:05d}.png"
                )

                if dataset_name in ["coco2017", "voc2012"]:
                    self.visualiser.visualise_instance_predictions(
                        image=image[0].numpy(),
                        predictions=instance_predictions,
                        fp=f"{self.dir_ckpt}/eval_images/{num_iter:05d}/{i:05d}_detectron2_instance_vis.png"
                    )

            if self.debug:
                break

        # save results
        if self.dir_ckpt is not None:
            results: dict = {"num_iter": num_iter, "timestamp": str(datetime.now())}
            results.update(scores[0])
            results.update(scores[1])

            if num_iter == iter_eval:
                json.dump(results, open(f"{self.dir_ckpt}/eval_metrics.json", 'w'))
            else:
                with open(f"{self.dir_ckpt}/eval_metrics.json", 'a') as f:
                    f.write('\n')
                    json.dump(results, f)
                    f.close()

            if dataset_name in ["coco2017", "voc2012"]:
                # save annotations
                [p.pop("bbox") for p in list_instance_predictions]
                json.dump(
                    list_instance_predictions,
                    open(f"{self.dir_ckpt}/instance_predictions_{num_iter:05d}.json", 'w'),
                    reject_bytes=False
                )

                try:
                    # compute coco-style metrics
                    coco_metrics: Dict[str, float] = self.compute_coco_metrics(
                        p_annotations=dataloader.dataset.p_annotations,
                        instance_predictions=list_instance_predictions,
                    )
                except IndexError:
                    # IndexError: list index out of range
                    coco_metrics: Dict[str, float] = {"index error": -1.0}

                # save coco-style metrics
                if num_iter == iter_eval:
                    json.dump(coco_metrics, open(f"{self.dir_ckpt}/eval_coco_style_metrics.json", 'w'))
                else:
                    with open(f"{self.dir_ckpt}/eval_coco_style_metrics.json", 'a') as f:
                        f.write('\n')
                        json.dump(coco_metrics, f)
                        f.close()

        if miou > self.best_miou and num_iter != -1:
            print(f"best mIoU is changed from {self.best_miou:.3f} to {miou:.3f}.")
            self.best_miou = miou

        torch.cuda.empty_cache()
