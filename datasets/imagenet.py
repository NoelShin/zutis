import os
from typing import Dict, List, Optional, Tuple
from glob import glob
import pickle as pkl
from itertools import chain
from copy import deepcopy
from random import randint, choices
import ujson as json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from pycocotools.mask import encode, decode
from utils.extract_text_embeddings import prompt_engineering
from utils.utils import get_network
from datasets.base_dataset import BaseDataset
from datasets.augmentations import copy_paste


class ImageNet1KDataset(BaseDataset):
    def __init__(
            self,
            dir_dataset: str,
            ignore_index: int,
            category_to_p_images_fp: str = None,
            categories: Optional[List[str]] = None,
            n_images: int = 500,
            max_n_masks: int = 10,
            clip_model_name: str = "ViT-L/14@336px",
            split: str = "train",
            scale_range: Optional[Tuple[float, float]] = (0.1, 1.0),
            crop_size: Optional[int] = 384,
            use_advanced_copy_paste: bool = False,
            max_n_partitions: int = 1,  # for advanced copy-paste  # 4
            min_distance: int = 48,  # for advanced copy-paste  # 48
            device: torch.device = torch.device("cuda:0"),
    ):
        super(ImageNet1KDataset, self).__init__()
        self.dir_dataset: str = dir_dataset
        self.ignore_index: int = ignore_index
        self.device: torch.device = device
        self.max_n_masks: int = max_n_masks
        self.use_advanced_copy_paste: bool = use_advanced_copy_paste

        category_to_p_images: Dict[str, List[str]] = self._get_p_images(
            category_to_p_images_fp=category_to_p_images_fp,
            n_images=n_images,
            categories=categories,
            clip_model_name=clip_model_name,
            split=split
        )

        # get a dictionary which will be used to assign a label id to a class-agnostic pseudo-mask.
        # note that for both pascal voc2012 and coco has a background category whose label id is 0.
        self.p_image_to_label_id: Dict[str, int] = {}
        for label_id, (category, p_images) in enumerate(category_to_p_images.items(), start=1):
            for p_image in p_images:
                self.p_image_to_label_id[p_image] = label_id

        self.p_images: List[str] = list(chain.from_iterable(category_to_p_images.values()))
        self.p_pseudo_masks: List[str] = self._get_pseudo_masks(
            dir_dataset=dir_dataset,
            p_images=self.p_images,
        )

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.scale_range = scale_range
        self.crop_size = crop_size
        self.split = split
        self.name: str = "imagenet"

        if self.use_advanced_copy_paste:
            from datasets.augmentations.advanced_copy_paste import AdvancedCopyPaste
            self.advanced_copy_paste = AdvancedCopyPaste(
                grid_size=crop_size, max_n_partitions=max_n_partitions, min_distance=min_distance
            )

    def _get_p_images(
            self,
            n_images: int,
            category_to_p_images_fp: str,
            categories: Optional[List[str]] = None,
            clip_model_name: str = "ViT-L/14@336px",
            split: str = "train"
    ) -> Dict[str, List[str]]:
        try:
            category_to_p_images: Dict[str, List[str]] = json.load(open(category_to_p_images_fp, 'r'))

        except FileNotFoundError:
            assert categories is not None, TypeError(categories)
            background_index = categories.index("background")
            if background_index != -1:
                categories.pop(background_index)
            assert "background" not in categories
            category_to_p_images = self.retrieve_images(
                n_images=n_images, categories=categories, clip_model_name=clip_model_name, split=split
            )
            json.dump(category_to_p_images, open(category_to_p_images_fp, 'w'))
            print(f"A category to image paths file is saved at {category_to_p_images_fp}.")
        return category_to_p_images

    def _convert_p_image_to_p_pseudo_mask(
            self,
            dir_dataset: str,
            p_image: str,
    ) -> str:
        split, wnid, filename = p_image.split('/')[-3:]
        dir_pseudo_mask: str = f"{dir_dataset}/{split}_pseudo_masks_selfmask/{wnid}"
        return f"{dir_pseudo_mask}/{filename.replace('JPEG', 'json')}"

    def _get_pseudo_masks(
            self, dir_dataset: str, p_images: List[str]
    ) -> List[str]:
        """
        Based on image paths and a dataset directory, return pseudo-masks for the images.
        If the system can't find a pseudo-mask for an image, we generate a pseudo-mask for the image and save it at a
        designated path.
        """
        p_pseudo_masks: List[str] = list()
        p_images_wo_pseudo_mask: List[str] = list()
        for p_image in tqdm(p_images):
            p_pseudo_mask: str = self._convert_p_image_to_p_pseudo_mask(
                dir_dataset=dir_dataset, p_image=p_image
            )
            p_pseudo_masks.append(p_pseudo_mask)
            if not os.path.exists(p_pseudo_mask):
                p_images_wo_pseudo_mask.append(p_image)

        if len(p_images_wo_pseudo_mask) > 0:
            print(f"Generating pseudo-masks for {len(p_images_wo_pseudo_mask)} images...")
            self.generate_pseudo_masks(p_images=p_images_wo_pseudo_mask, dir_dataset=dir_dataset)
        return p_pseudo_masks

    @torch.no_grad()
    def generate_pseudo_masks(
            self,
            p_images: List[str],
            dir_dataset: str,
            n_workers: int = 4,
            bilateral_solver: bool = True
    ) -> None:
        network = get_network(network_name="selfmask").to(self.device)
        network.eval()

        mask_dataset = MaskDataset(p_images=p_images)
        mask_dataloader = DataLoader(dataset=mask_dataset, batch_size=1, num_workers=n_workers, pin_memory=True)
        iter_mask_loader, pbar = iter(mask_dataloader), tqdm(range(len(mask_dataloader)))

        for _ in pbar:
            dict_data: dict = next(iter_mask_loader)
            image: torch.Tensor = dict_data["image"]  # 1 x 3 x H x W
            p_image: List[str] = dict_data["p_image"]  # 1

            try:
                dict_outputs: Dict[str, np.ndarray] = network(
                    image.to(self.device), inference=True, bilateral_solver=bilateral_solver
                )
            except RuntimeError:
                network.to("cpu")
                dict_outputs: Dict[str, np.ndarray] = network(image, inference=True, bilateral_solver=bilateral_solver)
                network.to(self.device)

            if bilateral_solver:
                dt: torch.Tensor = dict_outputs["dts_bi"][0]  # H x W, {0, 1}, torch.uint8
            else:
                dt: torch.Tensor = dict_outputs["dts"][0]  # H x W, {0, 1}, torch.uint8

            p_pseudo_mask = self._convert_p_image_to_p_pseudo_mask(dir_dataset=dir_dataset, p_image=p_image[0])
            os.makedirs(os.path.dirname(p_pseudo_mask), exist_ok=True)

            # Image.fromarray(dt_bi).save(p_pseudo_mask)
            # restore the original resolution before downsampling in the dataloader
            W, H = Image.open(p_image[0]).size
            dt: torch.Tensor = F.interpolate(dt[None, None], size=(H, W), mode="nearest")[0, 0]
            dt: np.ndarray = dt.cpu().numpy()

            rles: dict = encode(np.asfortranarray(dt))
            json.dump(rles, open(p_pseudo_mask, "w"), reject_bytes=False)

            # sanity check
            loaded_dt = decode(json.load(open(p_pseudo_mask, 'r')))
            assert (dt == loaded_dt).sum() == H * W

        print(f"Pseudo-masks are saved in {dir_dataset}.")

    def retrieve_images(
            self,
            categories: List[str],
            n_images: int = 100,
            clip_model_name: str = "ViT-L/14@336px",
            split: str = "train"
    ) -> Dict[str, List[str]]:
        """Retrieve images with CLIP"""
        assert split in ["train", "val"], ValueError(split)
        # extract text embeddings
        category_to_text_embedding: Dict[str, torch.Tensor] = prompt_engineering(
            model_name=clip_model_name, categories=categories
        )

        # len(categories) x n_dims, torch.float32, normalised
        text_embeddings: torch.Tensor = torch.stack(list(category_to_text_embedding.values()), dim=0)

        filename_to_img_embedding: dict = pkl.load(
            open(f"{self.dir_dataset}/filename_to_ViT_L_14_336px_{split}_img_embedding.pkl", "rb")
        )

        filenames: List[str] = list(filename_to_img_embedding.keys())

        # n_images x n_dims, torch.float32, normalised
        image_embeddings: torch.Tensor = torch.stack(list(filename_to_img_embedding.values()), dim=0).to(self.device)

        # compute cosine similarities between text and image embeddings
        similarities: torch.Tensor = text_embeddings @ image_embeddings.t()  # len(categories) x n_imgs

        category_to_p_images = dict()
        for category, category_similarities in zip(categories, similarities):
            indices: torch.Tensor = torch.argsort(category_similarities, descending=True)
            sorted_filenames: List[str] = np.array(filenames)[indices.cpu().tolist()].tolist()
            ret_filenames: List[str] = sorted_filenames[:n_images]  # topk retrieved images

            p_ret_imgs: List[str] = list()
            if split == "val":
                p_imgs: List[str] = sorted(glob(f"{self.dir_dataset}/val/**/*.JPEG"))
                filename_to_p_img: Dict[str, str] = dict()
                for p_img in p_imgs:
                    filename = os.path.basename(p_img)
                    filename_to_p_img[filename] = p_img

                for filename in ret_filenames:
                    p_img = filename_to_p_img[filename]
                    p_ret_imgs.append(p_img)
            else:
                for filename in ret_filenames:
                    wnid: str = filename.split('_')[0]
                    p_img: str = f"{self.dir_dataset}/train/{wnid}/{filename}"
                    p_ret_imgs.append(p_img)
            assert len(p_ret_imgs) > 0, ValueError(f"{len(p_ret_imgs)} == 0.")

            category_to_p_images[category] = p_ret_imgs

        return category_to_p_images

    def collate_fn(self, list_dict_data: List[dict]):
        batch_images: List[torch.Tensor] = list()
        batch_semantic_masks: List[torch.Tensor] = list()
        batch_instance_masks: List[torch.Tensor] = list()
        batch_category_ids: List[List[int]] = list()

        for dict_data in list_dict_data:
            batch_images.append(dict_data["image"])
            batch_semantic_masks.append(dict_data["semantic_mask"])
            batch_instance_masks.append(dict_data["instance_mask"])
            batch_category_ids.append(dict_data["category_ids"])

        return {
            "image": torch.stack(batch_images, dim=0),
            "semantic_mask": torch.stack(batch_semantic_masks, dim=0),
            "instance_mask": batch_instance_masks,
            "category_ids": batch_category_ids
        }

    def __len__(self):
        return len(self.p_images)

    def __getitem__(self, index: int) -> dict:
        dict_data: dict = {}

        images: List[torch.Tensor] = list()

        semantic_masks: List[torch.Tensor] = list()
        instance_masks: List[torch.Tensor] = list()

        if self.use_advanced_copy_paste:
            n_random_images: int = self.advanced_copy_paste.generate_grid()

            # randomly pick indices with repetition allowed
            random_indices: List[int] = choices(range(len(self.p_images)), k=n_random_images)

            random_images: List[torch.Tensor] = list()
            random_binary_masks: List[torch.Tensor] = list()
            category_ids: List[int] = list()
            for random_index in random_indices:
                p_image: str = self.p_images[random_index]
                image: Image.Image = Image.open(p_image).convert("RGB")
                # photometric augmentations
                image: Image.Image = self._photometric_augmentations(image)
                image: torch.Tensor = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)

                binary_mask: torch.Tensor = torch.from_numpy(
                    decode(json.load(open(self.p_pseudo_masks[random_index]))).astype(np.int64)
                )

                random_images.append(image)
                random_binary_masks.append(binary_mask)

                # assign a label id to a pseudo_mask
                label_id: int = self.p_image_to_label_id[p_image]
                category_ids.append(label_id)

            dict_copy_paste = self.advanced_copy_paste.copy_paste(
                images=random_images, binary_masks=random_binary_masks, category_ids=category_ids
            )

            dict_data.update({
                "image": dict_copy_paste["image"],
                "semantic_mask": dict_copy_paste["semantic_mask"],
                "instance_mask": dict_copy_paste["instance_mask"],
                "category_ids": category_ids,
            })

        else:
            n_masks = randint(1, self.max_n_masks)
            category_ids: List[int] = list()
            instance_ids: List[int] = list()
            for instance_id in range(1, n_masks + 1):  # start from 1
                instance_ids.append(instance_id)

                random_index = randint(0, len(self.p_images) - 1)
                p_image: str = self.p_images[random_index]
                p_pseudo_mask: str = self.p_pseudo_masks[random_index]

                image: Image.Image = Image.open(p_image).convert("RGB")
                binary_mask: np.ndarray = decode(json.load(open(p_pseudo_mask, 'r'))).astype(np.int64)

                image, _, binary_mask = self._geometric_augmentations(
                    image=image,
                    instance_mask=binary_mask,
                    ignore_index=self.ignore_index,
                    random_scale_range=(1.0, 1.0),
                    random_crop_size=384,
                    random_hflip_p=0.5
                )

                image: Image.Image = self._photometric_augmentations(image)  # 3 x crop_size x crop_size
                image: torch.Tensor = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)

                # assign a label id to a pseudo_mask
                label_id: int = self.p_image_to_label_id[p_image]
                category_ids.append(label_id)
                semantic_mask = deepcopy(binary_mask)
                semantic_mask[semantic_mask == 1] = label_id  # crop_size x crop_size, {0, label_id} or {0, label_id, 255}
                semantic_masks.append(semantic_mask)

                binary_mask[binary_mask == 1] = instance_id
                instance_masks.append(binary_mask)
                images.append(image)

            # overlaid_image: torch.Tensor (float32), 3 x crop_size x crop_size
            # overlaid_mask: torch.Tensor (int64), crop_size x crop_size
            overlaid_image, overlaid_semantic_mask, overlaid_instance_mask = copy_paste(
                images=images,
                semantic_masks=semantic_masks,
                instance_masks=instance_masks,
                background_index=0,
                ignore_index=self.ignore_index
            )

            try:
                # convert an instance mask to a one-hot mask
                instance_mask = torch.stack(
                    [overlaid_instance_mask == instance_id for instance_id in instance_ids], dim=0
                )
            except RuntimeError:
                assert self.split == "train"
                # RuntimeError: stack expects a non-empty TensorList
                instance_mask = torch.zeros((1, 384, 384))

            dict_data.update({
                "image": overlaid_image,
                "semantic_mask": overlaid_semantic_mask,
                "instance_mask": instance_mask,
                "category_ids": category_ids,
            })

        return dict_data


class MaskDataset(Dataset):
    def __init__(
            self,
            p_images: List[str],
            image_size: Optional[int] = 512,  # shorter side of image
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        assert len(p_images) > 0, f"No image paths are given: {len(p_images)}."
        self.p_images: List[str] = p_images
        self.image_size = image_size
        self.mean: Tuple[float, float, float] = mean
        self.std: Tuple[float, float, float] = std

    def __len__(self):
        return len(self.p_images)

    def __getitem__(self, index: int) -> dict:
        image_path: str = self.p_images[index]
        image: Image.Image = Image.open(image_path).convert("RGB")
        if self.image_size is not None:
            image = TF.resize(image, size=self.image_size, interpolation=Image.BILINEAR)
        image = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)
        return {"image": image, "p_image": image_path}
