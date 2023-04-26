import os
from typing import Dict, List, Optional, Tuple
from glob import glob
import pickle as pkl
from itertools import chain

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
from utils.extract_image_embeddings import extract_image_embeddings
from utils.utils import get_network


class PASS:
    def __init__(
            self,
            dir_dataset: str,
            clip_model_name: str = "ViT-L/14@336px",
            category_to_p_images_fp: str = None,
            categories: Optional[List[str]] = None,
            n_images: int = 500,
            device: torch.device = torch.device("cuda:0"),
    ):
        # 1,439,588 jpg images
        self.dir_dataset: str = dir_dataset
        self.clip_model_name: str = clip_model_name
        self.device: torch.device = device

        self.p_images: List[str] = sorted(glob(f"{dir_dataset}/images/*.jpg"))

        category_to_p_images: Dict[str, List[str]] = self._get_p_images(
            category_to_p_images_fp=category_to_p_images_fp,
            n_images=n_images,
            categories=categories,
            clip_model_name=clip_model_name
        )

        # get a dictionary which will be used to assign a label id to a class-agnostic pseudo-mask.
        # note that for both pascal voc2012 and coco has a background category whose label id is 0.
        self.p_image_to_label_id: Dict[str, int] = {}
        for label_id, (category, p_images) in enumerate(category_to_p_images.items(), start=1):
            for p_image in p_images:
                self.p_image_to_label_id[p_image] = label_id

        # update self.p_images with the image paths retrieved by CLIP
        self.p_images: List[str] = list(chain.from_iterable(category_to_p_images.values()))
        self.p_pseudo_masks: List[str] = self._get_pseudo_masks(
            dir_dataset=dir_dataset,
            p_images=self.p_images,
        )
        self.name: str = "pass"

    def _get_p_images(
            self,
            n_images: int,
            category_to_p_images_fp: str,
            categories: Optional[List[str]] = None,
            clip_model_name: str = "ViT-L/14@336px",
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
                n_images=n_images, categories=categories, clip_model_name=clip_model_name
            )
            json.dump(category_to_p_images, open(category_to_p_images_fp, 'w'))
            print(f"A category to image paths file is saved at {category_to_p_images_fp}.")
        return category_to_p_images

    def retrieve_images(
            self,
            categories: List[str],
            n_images: int = 100,
            clip_model_name: str = "ViT-L/14@336px",
    ) -> Dict[str, List[str]]:
        """Retrieve images with CLIP"""
        # extract text embeddings
        category_to_text_embedding: Dict[str, torch.Tensor] = prompt_engineering(
            model_name=clip_model_name, categories=categories
        )

        # len(categories) x n_dims, torch.float32, normalised
        text_embeddings: torch.Tensor = torch.stack(list(category_to_text_embedding.values()), dim=0)

        _clip_model_name = clip_model_name.replace('-', '_').replace('/', '_').replace('@', '_')
        p_filename_to_img_embedding: str = f"{self.dir_dataset}/filename_to_{_clip_model_name}_img_embedding.pkl"
        try:
            filename_to_img_embedding: dict = pkl.load(
                open(p_filename_to_img_embedding, "rb")
            )
        except FileNotFoundError:
            # extract image embeddings
            filename_to_img_embedding: dict = extract_image_embeddings(
                model_name=clip_model_name, p_images=self.p_images, fp=p_filename_to_img_embedding
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

            for filename in ret_filenames:
                p_img: str = f"{self.dir_dataset}/images/{filename}"
                p_ret_imgs.append(p_img)
            assert len(p_ret_imgs) > 0, ValueError(f"{len(p_ret_imgs)} == 0.")

            category_to_p_images[category] = p_ret_imgs
        return category_to_p_images

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

    def _convert_p_image_to_p_pseudo_mask(
            self,
            dir_dataset: str,
            p_image: str,
    ) -> str:
        filename = p_image.split('/')[-1]
        ext = filename.split('.')[-1]
        dir_pseudo_mask: str = f"{dir_dataset}/pseudo_masks_selfmask"
        return f"{dir_pseudo_mask}/{filename.replace(ext, 'json')}"

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
            p_pseudo_mask: str = self._convert_p_image_to_p_pseudo_mask(dir_dataset=dir_dataset, p_image=p_image)
            p_pseudo_masks.append(p_pseudo_mask)
            if not os.path.exists(p_pseudo_mask):
                p_images_wo_pseudo_mask.append(p_image)

        if len(p_images_wo_pseudo_mask) > 0:
            print(f"Generating pseudo-masks for {len(p_images_wo_pseudo_mask)} images...")
            self.generate_pseudo_masks(p_images=p_images_wo_pseudo_mask, dir_dataset=dir_dataset)
        return p_pseudo_masks

    def __len__(self) -> int:
        return len(self.p_images)

    def __getitem__(self, index: int):
        return


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
