from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union
import random
from random import seed, shuffle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import colorsys


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataset(
        dir_dataset: Union[str, List[str]],
        dataset_name: str,
        p_filename_to_image_embedding: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
        image_size: Optional[int] = None,
        ignore_index: Optional[int] = None,
        categories: Optional[List[str]] = None,
        category_to_p_images_fp: Optional[str] = None,
        n_images: int = 500,
        scale_range: Tuple[float, float] = (0.1, 1.0),
        use_advanced_copy_paste: bool = False,
        n_categories: Optional[int] = 919,  # for ImageNet-S benchmarks
        random_duplicate: Optional[bool] = False,
        **dataloader_kwargs
) -> Union[Dataset, DataLoader]:
    if dataset_name == "imagenet":
        from datasets import ImageNet1KDataset
        dataset = ImageNet1KDataset(
            dir_dataset=dir_dataset,
            split=split,
            ignore_index=ignore_index,
            categories=categories,
            category_to_p_images_fp=category_to_p_images_fp,
            n_images=n_images,
            scale_range=scale_range,
            crop_size=image_size,
            use_advanced_copy_paste=use_advanced_copy_paste
        )
    elif dataset_name == "index":
        from datasets import IndexDataset

        dataset = IndexDataset(
            dir_dataset=dir_dataset,
            ignore_index=ignore_index,
            p_filename_to_image_embedding=p_filename_to_image_embedding,
            categories=categories,
            category_to_p_images_fp=category_to_p_images_fp,
            n_images=n_images,
            scale_range=scale_range,
            crop_size=image_size,
            random_duplicate=random_duplicate
        )
    elif dataset_name == "voc2012":
        from datasets import VOC2012Dataset
        dataset = VOC2012Dataset(dir_dataset=dir_dataset, split=split)

    elif dataset_name == "coco2017":
        from datasets.coco2017 import COCO2017Dataset
        dataset = COCO2017Dataset(dir_dataset=dir_dataset, split=split)

    elif dataset_name == "coca":
        from datasets.coca import COCADataset
        dataset = COCADataset(dir_dataset=dir_dataset)

    elif "imagenet-s" in dataset_name:
        from datasets.imagenet_s import ImageNetSDataset
        dataset = ImageNetSDataset(
            dir_dataset=dir_dataset,
            split=split,
            n_categories=n_categories
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name} (choose among imagenet, voc2012, imagenet-s)")

    if dataloader_kwargs is not None:
        return DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn if dataset_name in [
                "voc2012", "coco2017", "index", "imagenet"
            ] else None,
            **dataloader_kwargs
        )
    else:
        return dataset


def get_experim_name(args: Namespace) -> str:
    kwargs: List[str] = [args.clip_arch.lower().replace('/', '_').replace('-', '_').replace('@', '_')]

    if "RN" in args.clip_arch:
        try:
            if args.frozen_bn:
                kwargs.append("frozen_bn")
        except AttributeError:
            pass

    if args.dataset_name == "voc2012":
        # ablation study
        bs = args.train_dataloader_kwargs["batch_size"]
        kwargs.append(f"bs{bs}")

    try:
        stop_gradient = args.stop_gradient
        if not stop_gradient:
            kwargs.append("no_sg")  # no stop-gradient
    except AttributeError:
        pass

    try:
        random_duplicate = args.random_duplicate
        if random_duplicate:
            kwargs.append("rd")  # random duplicate
    except AttributeError:
        pass

    if args.index_dataset_name == "index":
        kwargs.append(f"n{args.n_images}")
        for p_train_dataset in args.dir_train_dataset:
            dir_name = p_train_dataset.split('/')[-2]
            if dir_name == "ImageNet2012":
                kwargs.append("imagenet")
            elif dir_name == "pass":
                kwargs.append("pass")
            else:
                raise ValueError(dir_name)

    kwargs.append(f"sr{int(args.scale_range[0] * 100)}{int(args.scale_range[1] * 100)}")

    if args.suffix != '':
        kwargs.append(args.suffix)

    # seed number
    kwargs.append(f"s{args.seed}")

    if args.debug:
        kwargs.append("debug")
    return '_'.join(kwargs)


def get_network(
        network_name: str,
        encoder_type: Optional[str] = "clip",
        categories: Optional[List[str]] = None,
        frozen_bn: Optional[bool] = True,
        stop_gradient: Optional[bool] = True,
        decoder_image_n_dims: Optional[int] = None,
) -> torch.nn.Module:
    if network_name == "selfmask":
        from networks.selfmask.selfmask import SelfMask
        network = SelfMask()
        state_dict = torch.hub.load_state_dict_from_url(
            "https://www.robots.ox.ac.uk/~vgg/research/selfmask/shared_files/selfmask_nq20.pt"
        )
        network.load_state_dict(state_dict=state_dict, strict=True)

    else:
        assert network_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "RN50", "RN50x4", "RN50x16", "RN50x64", "RN101", "dilatedRN50"]
        from networks.zutis import ZUTIS
        network = ZUTIS(
            clip_arch=network_name,
            encoder_type=encoder_type,
            categories=categories,
            frozen_bn=frozen_bn,
            stop_gradient=stop_gradient,
            decoder_image_n_dims=decoder_image_n_dims
        )
    print(f"{network_name} is loaded.")
    return network


def get_optimiser(
        network: torch.nn.Module,
):
    base_lr: float = 5e-5
    params = [
        {'params': network.encoder.parameters(), "lr": base_lr * 0.1, "weight_decay": 0.05},
        {'params': network.decoder.parameters(), "lr": base_lr, "weight_decay": 0.05},
        {'params': network.query_embed, "lr": base_lr, "weight_decay": 0.05},
        {'params': network.ffn1.parameters(), "lr": base_lr, "weight_decay": 0.05},
        {'params': network.ffn2.parameters(), "lr": base_lr, "weight_decay": 0.05},
    ]

    optimiser = torch.optim.AdamW(params=params)
    return optimiser


def get_lr_scheduler(optimiser: torch.optim.Optimizer, n_iters: int):
    from utils.scheduler import PolyLR
    return PolyLR(optimiser, n_iters, power=0.9)


def get_palette(
        dataset_name: str,
        n_categories: Optional[int] = None
) -> Dict[int, Union[Tuple[int, int, int], Tuple[float, float, float]]]:
    if "voc2012" in dataset_name:
        from datasets.voc2012 import voc2012_palette
        palette: Dict[int, Tuple[int, int, int]] = voc2012_palette

    elif dataset_name in ["coco2017"]:
        from datasets.coco2017 import coco2017_palette
        palette = coco2017_palette
    elif dataset_name == "coca":
        from datasets.coca import coca_palette
        palette = coca_palette
    elif dataset_name == "imagenet-s":
        if n_categories == 50:
            from datasets.imagenet_s import imagenet_s50_palette
            palette = imagenet_s50_palette
        elif n_categories == 300:
            from datasets.imagenet_s import imagenet_s300_palette
            palette = imagenet_s300_palette
        elif n_categories == 919:
            from datasets.imagenet_s import imagenet_s919_palette
            palette = imagenet_s919_palette
    else:
        raise ValueError(dataset_name)
    return palette


def get_label_id_to_category(
        dataset_name: str, n_categories: Optional[int] = 919,
) -> Dict[int, str]:
    if "voc2012" in dataset_name:
        from datasets.voc2012 import label_id_to_category
        label_id_to_category: Dict[int, str] = label_id_to_category

    elif dataset_name == "coco2017":
        from datasets.coco2017 import label_id_to_category
        label_id_to_category: Dict[int, str] = label_id_to_category

    elif dataset_name == "coca":
        from datasets.coca import label_id_to_category
        label_id_to_category: Dict[int, str] = label_id_to_category

    elif dataset_name == "imagenet-s":
        if n_categories == 50:
            from datasets.imagenet_s import label_id_to_category_50
            label_id_to_category: Dict[int, str] = label_id_to_category_50
        elif n_categories == 300:
            from datasets.imagenet_s import label_id_to_category_300
            label_id_to_category: Dict[int, str] = label_id_to_category_300
        elif n_categories == 919:
            from datasets.imagenet_s import label_id_to_category_919
            label_id_to_category: Dict[int, str] = label_id_to_category_919
        else:
            raise ValueError(n_categories)
    else:
        raise ValueError(dataset_name)
    return label_id_to_category


def convert_tensor_to_pil_image(
        tensor: torch.Tensor,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Image.Image:
    assert len(tensor.shape) == 3, ValueError(f"{tensor.shape}")

    # 3 x H x W
    tensor = tensor * torch.tensor(std, device=tensor.device)[:, None, None]
    tensor = tensor + torch.tensor(mean, device=tensor.device)[:, None, None]
    tensor = torch.clip(tensor * 255, 0, 255)
    pil_image: Image.Image = Image.fromarray(tensor.cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
    return pil_image


def colourise_mask(
        mask: np.ndarray,
        palette: Union[List[Tuple[int, int, int]], Dict[int, Tuple[int, int, int]]],
):
    assert len(mask.shape) == 2, ValueError(mask.shape)
    h, w = mask.shape
    grid = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = set(mask.flatten())
    for l in unique_labels:
        try:
            grid[mask == l] = np.array(palette[l], dtype=np.uint8)
        except IndexError:
            raise IndexError(f"No colour is found for a label id: {l}")
    return grid


def load_model(model, arch: str, patch_size: int) -> None:
    url = None
    if arch == "deit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif arch == "deit_small" and patch_size == 8:
        # model used for visualizations in our paper
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    elif arch == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif arch == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("There is no reference weights available for this model => We use random weights.")


def getDistinctColors(n):
    def HSVToRGB(h, s, v):
        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        return int(255 * r), int(255 * g), int(255 * b)

    seed(0)
    indices = list(range(0, n))
    shuffle(indices)

    huePartition = 1.0 / (n + 1)
    return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in indices]
