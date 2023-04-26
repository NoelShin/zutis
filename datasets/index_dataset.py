import os
from typing import Dict, List, Optional, Tuple, Union
import pickle as pkl
from itertools import chain
from copy import deepcopy
from random import randint, random, choice
from time import time
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


class IndexDataset(BaseDataset):
    def __init__(
            self,
            dir_dataset: Union[str, List[str]],
            p_filename_to_image_embedding: Union[str, List[str]],
            ignore_index: int,
            clip_model_name: str = "ViT-L/14@336px",
            categories: Optional[List[str]] = None,
            n_images: int = 500,
            category_to_p_images_fp: Optional[str] = None,
            device: torch.device = torch.device("cuda:0"),
            max_n_masks: int = 10,
            scale_range: Optional[Tuple[float, float]] = (0.1, 1.0),
            crop_size: Optional[int] = 384,
            random_duplicate: bool = False
    ):
        super(IndexDataset, self).__init__()
        self.dir_dataset: str = dir_dataset
        self.clip_model_name: str = clip_model_name
        self.device: torch.device = device
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.ignore_index: int = ignore_index
        self.max_n_masks: int = max_n_masks
        self.scale_range: Optional[Tuple[float, float]] = scale_range
        self.crop_size: Optional[int] = crop_size
        self.name: str = "index"
        self.random_duplicate: bool = random_duplicate

        category_to_p_images: Dict[str, List[str]] = self._get_category_to_p_images(
            p_filename_to_image_embedding=p_filename_to_image_embedding,
            clip_model_name=clip_model_name,
            categories=categories,
            dir_dataset=dir_dataset,
            n_images=n_images,
            category_to_p_images_fp=category_to_p_images_fp
        )

        # get a dictionary which will be used to assign a label id to a class-agnostic pseudo-mask.
        # note that for both pascal voc2012 and coco has a background category whose label id is 0.
        self.p_image_to_label_id: Dict[str, int] = {}

        if categories[0] == "background":
            categories = categories[1:]

        for label_id, category in enumerate(categories, start=1):
            # if label_id == 0:
            #     assert category == "background", ValueError(category)
            #     continue

            p_images = category_to_p_images[category]
            for p_image in p_images:
                self.p_image_to_label_id[p_image] = label_id

        # for label_id, (category, p_images) in enumerate(category_to_p_images.items(), start=1):
        #     for p_image in p_images:
        #         self.p_image_to_label_id[p_image] = label_id

        # update self.p_images with the image paths retrieved by CLIP
        self.p_images: List[str] = list(chain.from_iterable(category_to_p_images.values()))
        self.p_pseudo_masks: List[str] = self._get_pseudo_masks(
            dir_dataset=dir_dataset,
            p_images=self.p_images,
        )
        self.p_image_to_p_pseudo_mask: Dict[str, str] = {
            p_image: p_pseudo_mask for p_image, p_pseudo_mask in zip(self.p_images, self.p_pseudo_masks)
        }
        self.category_to_p_images: Dict[str, List[str]] = category_to_p_images

        # self.p_image_to_index: Dict[str, int] = {p_image: index for index, p_image in enumerate(self.p_images)}
        # self.p_pseudo_mask_to_index: Dict[str, int] = {}

    def _get_category_to_p_images(
            self,
            p_filename_to_image_embedding: Union[str, List[str]],
            clip_model_name: str,
            categories: List[str],
            dir_dataset: Union[str, List[str]],
            n_images: int,
            category_to_p_images_fp: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        try:
            category_to_p_images = json.load(open(category_to_p_images_fp, "r"))

        except (TypeError, FileNotFoundError):
            print("Generating category_to_p_images...")

            # exclude background category if it exists (for retrieval, we do not consider background images)
            try:
                background_index = categories.index("background")
                categories = deepcopy(categories)
                categories.pop(background_index)
            except ValueError:
                pass

            # if background_index != -1:
            #     categories = deepcopy(categories)
            #     categories.pop(background_index)
            # assert "background" not in categories

            # extract text embeddings
            category_to_text_embedding: Dict[str, torch.Tensor] = prompt_engineering(
                model_name=clip_model_name, categories=categories
            )

            # len(categories) x n_dims, torch.float32, normalised
            text_embeddings: torch.Tensor = torch.stack(
                list(category_to_text_embedding.values()), dim=0
            ).to(self.device)
            if not isinstance(dir_dataset, list):
                dir_dataset: List[str] = [dir_dataset]
                p_filename_to_image_embedding: List[str] = [p_filename_to_image_embedding]

            assert len(p_filename_to_image_embedding) == len(dir_dataset), \
                f"{len(p_filename_to_image_embedding)} != {len(dir_dataset)}"

            p_images: List[str] = list()
            image_embeddings: List[torch.Tensor] = list()
            for d, p in zip(dir_dataset, p_filename_to_image_embedding):
                st = time()
                filename_to_image_embedding: dict = pkl.load(open(p, "rb"))
                print(f"(Loading) {time() - st:.3f} sec.")
                if "ImageNet2012" in d:
                    _p_images: list = list()
                    for filename in tqdm(filename_to_image_embedding.keys()):
                        wnid = filename.split('_')[0]
                        _p_image: str = f"{d}/{wnid}/{filename}"
                        _p_images.append(_p_image)
                    p_images.extend(_p_images)
                else:
                    _p_images: list = list()
                    for filename in tqdm(filename_to_image_embedding.keys()):
                        _p_image: str = f"{d}/{filename}"
                        _p_images.append(_p_image)
                    p_images.extend(_p_images)
                image_embeddings.extend(list(filename_to_image_embedding.values()))
            image_embeddings: torch.Tensor = torch.stack(image_embeddings, dim=0).to(self.device)  # n_imgs x n_dims

            category_to_p_images = dict()

            # compute cosine similarities between text and image embeddings
            category_similarities: torch.Tensor = text_embeddings @ image_embeddings.t()  # len(categories) x n_imgs
            for category, similarities in zip(categories, category_similarities):
                indices: torch.Tensor = torch.argsort(similarities, descending=True)
                sorted_p_images: List[str] = np.array(p_images)[indices.cpu().tolist()].tolist()
                category_to_p_images[category] = sorted_p_images[:n_images]  # topk retrieved images

            if category_to_p_images_fp is not None:
                if not os.path.exists(os.path.dirname(category_to_p_images_fp)):
                    os.makedirs(os.path.dirname(category_to_p_images_fp), exist_ok=True)
                    print(f"{os.path.dirname(category_to_p_images_fp)} is created.")
                json.dump(category_to_p_images, open(category_to_p_images_fp, "w"))
                print(f"category_to_p_images is saved at {category_to_p_images_fp}.")
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

            p_pseudo_mask = self._convert_p_image_to_p_pseudo_mask(p_image=p_image[0])
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
            p_image: str,
            self_training: bool = False,
            dir_ckpt: Optional[str] = None
    ) -> str:
        if self_training:
            assert dir_ckpt is not None
            os.makedirs(f"{dir_ckpt}/self_training", exist_ok=True)
            filename = p_image.split('/')[-1]
            ext = filename.split('.')[-1]
            return f"{dir_ckpt}/self_training/{filename.replace(ext, 'json')}"

        else:
            if "/ImageNet2012/" in p_image:
                dir_dataset = '/'.join(p_image.split('/')[:-3])
                split, wnid, filename = p_image.split('/')[-3:]
                dir_pseudo_mask: str = f"{dir_dataset}/{split}_pseudo_masks_selfmask/{wnid}"
                return f"{dir_pseudo_mask}/{filename.replace('JPEG', 'json')}"

            elif "/pass/" in p_image:
                dir_dataset = '/'.join(p_image.split('/')[:-1])
                filename = p_image.split('/')[-1]
                ext = filename.split('.')[-1]
                dir_pseudo_mask: str = f"{dir_dataset.replace('/images', '')}/pseudo_masks_selfmask"
                return f"{dir_pseudo_mask}/{filename.replace(ext, 'json')}"
            else:
                raise ValueError(p_image)

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
            p_pseudo_mask: str = self._convert_p_image_to_p_pseudo_mask(p_image=p_image)
            p_pseudo_masks.append(p_pseudo_mask)
            if not os.path.exists(p_pseudo_mask):
                p_images_wo_pseudo_mask.append(p_image)

        if len(p_images_wo_pseudo_mask) > 0:
            print(f"Generating pseudo-masks for {len(p_images_wo_pseudo_mask)} images...")
            self.generate_pseudo_masks(p_images=p_images_wo_pseudo_mask, dir_dataset=dir_dataset)
        return p_pseudo_masks

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

        n_masks = randint(1, self.max_n_masks)
        category_ids: List[int] = list()
        instance_ids: List[int] = list()
        list_random_indices: List[int] = list()

        random_category: Optional[str] = None
        if self.random_duplicate and random() > 0.5:
            random_category: str = choice(list(self.category_to_p_images.keys()))
            assert random_category != "background", ValueError(random_category)

        for instance_id in range(1, n_masks + 1):
            instance_ids.append(instance_id)

            if random_category is not None:
                p_category_images = self.category_to_p_images[random_category]
                random_index = randint(0, len(p_category_images) - 1)
                p_image: str = p_category_images[random_index]
                p_pseudo_mask: str = self.p_image_to_p_pseudo_mask[p_image]
            else:
                p_images = self.p_images
                random_index = randint(0, len(p_images) - 1)
                p_image: str = self.p_images[random_index]
                p_pseudo_mask: str = self.p_pseudo_masks[random_index]

            image: Image.Image = Image.open(p_image).convert("RGB")
            binary_mask: np.ndarray = decode(json.load(open(p_pseudo_mask, 'r'))).astype(np.int64)

            image, _, binary_mask = self._geometric_augmentations(
                image=image,
                instance_mask=binary_mask,
                ignore_index=self.ignore_index,
                random_scale_range=self.scale_range,
                random_crop_size=self.crop_size,
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
