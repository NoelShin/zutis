from typing import Optional, Tuple, Union
from random import randint, random, uniform
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode as IM


def random_crop(
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        crop_size: Union[int, Tuple[int, int]],  # (h, w)
        fill: Union[int, Tuple[int, int, int]],  # an unsigned integer or RGB,
        padding: Optional[Tuple[int, int, int, int]] = None,  # (left, top, right, bottom)
        offset: Optional[Tuple[int, int]] = None,  # (top, left) coordinate of a crop
):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    else:
        assert isinstance(crop_size, (tuple, list)) and len(crop_size) == 2

    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
        h, w = image.shape[-2:]
    elif isinstance(image, Image.Image):
        w, h = image.size
    elif isinstance(image, torch.Tensor):
        h, w = image.shape[-2:]
    else:
        raise TypeError(type(image))

    pad_h, pad_w = max(crop_size[0] - h, 0), max(crop_size[1] - w, 0)

    if padding is None:
        if random() < 0.25:
            padding = [pad_w, pad_h, 0, 0]  # left, top, right, bottom
        elif 0.25 <= random() < 0.5:
            padding = [pad_w, 0, 0, pad_h]  # left, top, right, bottom
        elif 0.5 <= random() < 0.75:
            padding = [0, pad_h, pad_w, 0]  # left, top, right, bottom
        else:
            padding = [0, 0, pad_w, pad_h]  # left, top, right, bottom
    image = TF.pad(image, padding, fill=fill, padding_mode="constant")

    if isinstance(image, Image.Image):
        w, h = image.size
    else:
        h, w = image.shape[-2:]

    if offset is None:
        offset = (randint(0, h - crop_size[0]), randint(0, w - crop_size[1]))

    image = TF.crop(image, top=offset[0], left=offset[1], height=crop_size[0], width=crop_size[1])
    return image, padding, offset


def compute_size(
        input_size: Tuple[int, int],  # h, w
        output_size: int,
        edge: str
) -> Tuple[int, int]:
    assert edge in ["shorter", "longer"]
    h, w = input_size

    if edge == "longer":
        if w > h:
            h = int(float(h) / w * output_size)
            w = output_size
        else:
            w = int(float(w) / h * output_size)
            h = output_size
        assert w <= output_size and h <= output_size

    else:
        if w > h:
            w = int(float(w) / h * output_size)
            h = output_size
        else:
            h = int(float(h) / w * output_size)
            w = output_size
        assert w >= output_size and h >= output_size
    return h, w


def resize(
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        size: Union[int, Tuple[int, int]],
        interpolation: str,
        edge: str = "shorter",
        max_size: Optional[int] = None
) -> Union[Image.Image, torch.Tensor]:
    """
    :param image: an image to be resized
    :param size: a resulting image size
    :param interpolation: sampling mode. ["nearest", "bilinear", "bicubic"]
    :param edge: Default: "shorter"
    :param max_size: Default: None

    No-op if a size is given as a tuple (h, w).
    If set to "both", resize both height and width to the specified size.
    If set to "shorter", resize the shorter edge to the specified size keeping the aspect ratio.
    If set to "longer", resize the longer edge to the specified size keeping the aspect ratio.
    :return: a resized image
    """
    assert interpolation in ["nearest", "bilinear", "bicubic"], ValueError(interpolation)
    assert edge in ["both", "shorter", "longer"], ValueError(edge)
    interpolation = {"nearest": IM.NEAREST, "bilinear": IM.BILINEAR, "bicubic": IM.BICUBIC}[interpolation]

    if type(image) == torch.Tensor:
        image = image.clone().detach()
    elif type(image) == np.ndarray:
        image = torch.from_numpy(image)

    if type(size) is tuple:
        if type(image) == torch.Tensor and len(image.shape) == 2:
            image = TF.resize(
                image.unsqueeze(dim=0), size=size, interpolation=interpolation, max_size=max_size
            ).squeeze(dim=0)
        else:
            image = TF.resize(image, size=size, interpolation=interpolation, max_size=max_size)

    else:
        if edge == "both":
            if len(image.shape) == 2:
                image = image[None, None]
                image = TF.resize(image, size=[size, size], interpolation=interpolation)[0, 0]
            elif len(image.shape) == 3:
                image = image[None]
                image = TF.resize(image, size=[size, size], interpolation=interpolation)[0]
            elif len(image.shape) == 4:
                image = TF.resize(image, size=[size, size], interpolation=interpolation)
            else:
                raise ValueError(f"{len(image.shape)} != 3 or 4.")

        else:
            if edge == "shorter":
                assert isinstance(size, int), f"{type(size)} is not int"
                if len(image.shape) == 2:
                    image = image[None, None]
                    image = TF.resize(image, size=size, interpolation=interpolation, max_size=max_size)[0, 0]
                elif len(image.shape) == 3:
                    image = image[None]
                    image = TF.resize(image, size=size, interpolation=interpolation, max_size=max_size)[0]
                elif len(image.shape) == 4:
                    image = TF.resize(image, size=size, interpolation=interpolation, max_size=max_size)
                else:
                    raise ValueError(f"{len(image.shape)} != 3 or 4.")
            else:
                # edge == "longer"
                # TF.resize function does not provide a case you want to set an integer for a longer side.
                # For this end, an output size is computed and given to the TF.resize function.
                if isinstance(image, Image.Image):
                    w, h = image.size
                else:
                    h, w = image.shape[-2:]
                rh, rw = compute_size(input_size=(h, w), output_size=size, edge=edge)

                if isinstance(image, Image.Image):
                    image = TF.resize(image, size=[rh, rw], interpolation=interpolation, max_size=max_size)
                else:
                    if len(image.shape) == 2:
                        image = image[None, None]
                        image = TF.resize(image, size=[rh, rw], interpolation=interpolation, max_size=max_size)[0, 0]
                    elif len(image.shape) == 3:
                        image = image[None]
                        image = TF.resize(image, size=[rh, rw], interpolation=interpolation, max_size=max_size)[0]
                    elif len(image.shape) == 4:
                        image = TF.resize(image, size=[rh, rw], interpolation=interpolation, max_size=max_size)
                    else:
                        raise ValueError(f"{len(image.shape)} != 3 or 4.")
    return image


def random_scale(
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        random_scale_range: Tuple[float, float],
        semantic_mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
        instance_mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None
):
    scale = uniform(*random_scale_range)
    if isinstance(image, Image.Image):
        w, h = image.size
    else:
        h, w = image.shape[-2:]
    w_rs, h_rs = int(w * scale), int(h * scale)
    image: Image.Image = resize(image, size=(h_rs, w_rs), interpolation="bilinear")
    if semantic_mask is not None:
        semantic_mask = resize(semantic_mask, size=(h_rs, w_rs), interpolation="nearest")

    if instance_mask is not None:
        instance_mask = resize(instance_mask, size=(h_rs, w_rs), interpolation="nearest")
    return image, semantic_mask, instance_mask


def random_hflip(
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        p: float,
        semantic_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        instance_mask: Optional[Union[np.ndarray, torch.Tensor]] = None

):
    assert 0. <= p <= 1., ValueError(random_hflip)
    # Return a random floating point number in the range [0.0, 1.0).
    if random() > p:
        image = TF.hflip(image)
        if semantic_mask is not None:
            semantic_mask = TF.hflip(semantic_mask)

        if instance_mask is not None:
            instance_mask = TF.hflip(instance_mask)
    return image, semantic_mask, instance_mask
