from typing import Dict, Iterable, List, Optional, Tuple, Union
from copy import deepcopy
from random import choice, randint, sample
import torch
from datasets.augmentations.geometric_transforms import random_crop, resize
from datasets.augmentations.copy_paste import mask_to_bbox


class AdvancedCopyPaste:
    def __init__(
            self,
            grid_size: Union[int, Tuple[int, int]],
            max_n_partitions: int = 9,
            min_distance: int = 10
    ):
        self.grid_size: Union[int, Tuple[int, int]] = grid_size
        self.max_n_partitions: int = max_n_partitions
        self.min_distance: int = min_distance

        self.patch_info: Dict[int, Dict[str, int]] = None
        self.n_random_images: int = None
        self.random_patch_indices: List[int] = None

    def get_n_random_images(self) -> int:
        assert self.n_random_images is not None
        return self.n_random_images

    def get_max_n_partitions(self) -> int:
        return self.max_n_partitions

    def set_max_n_partitions(self, max_n_partitions: int) -> None:
        self.max_n_partitions = max_n_partitions

    @staticmethod
    def select_new_bar(prev_bars: Iterable, min_distance: int, max_length: int, verbose: bool = False) -> int:
        new_bar: int
        occupied_regions: set = set()

        for bar in prev_bars:
            occupied_regions.update(range(bar - min_distance, bar + min_distance))

        candidate_regions = set(range(max_length)) - occupied_regions
        try:
            new_bar = choice(list(candidate_regions))
        except IndexError:
            # IndexError: Cannot choose from an empty sequence
            if verbose:
                print("all regions are occupied.")
            new_bar = 0
        return new_bar

    def partition_grid(
            self,
            grid_size: Union[int, Tuple[int, int]],
            n_partitions: Union[int, Tuple[int, int]],
            min_distance: int,
    ) -> Dict[int, Dict[str, int]]:
        if isinstance(grid_size, int):
            h = w = grid_size
        else:
            assert len(grid_size) == 2
            h, w = grid_size

        if isinstance(n_partitions, int):
            n_partitions_x = n_partitions_y = n_partitions
        else:
            assert len(n_partitions) == 2
            n_partitions_x, n_partitions_y = n_partitions

        vertical_bars: set = {0, w + 1}
        horizontal_bars: set = {0, h + 1}

        for i in range(n_partitions_x):
            vertical_bars.update([
                self.select_new_bar(prev_bars=vertical_bars, min_distance=min_distance, max_length=w)
            ])
        vertical_bars.remove(w + 1)
        vertical_bars.update([w])
        vertical_bars: list = sorted(vertical_bars)

        for i in range(n_partitions_y):
            horizontal_bars.update([
                self.select_new_bar(prev_bars=horizontal_bars, min_distance=min_distance, max_length=h)
            ])
        horizontal_bars.remove(h + 1)
        horizontal_bars.update([h])
        horizontal_bars: list = sorted(horizontal_bars)

        patch_info: Dict[int, Dict[str, int]] = dict()
        patch_index: int = 0
        for i in range(len(horizontal_bars) - 1):
            h_bar = horizontal_bars[i]
            height = horizontal_bars[i + 1] - h_bar
            for j in range(len(vertical_bars) - 1):
                v_bar = vertical_bars[j]
                width = vertical_bars[j + 1] - v_bar
                patch_info[patch_index] = {"top": h_bar, "left": v_bar, "width": width, "height": height}
                patch_index += 1
        return patch_info

    def generate_grid(self) -> int:
        if self.max_n_partitions == 1:
            self.n_random_images = 0
            self.random_patch_indices = []
        else:
            n_partitions_x = choice(range(1, self.max_n_partitions))
            n_partitions_y = choice(range(1, self.max_n_partitions))
            self.patch_info: Dict[int, Dict[str, int]] = self.partition_grid(
                grid_size=self.grid_size,
                n_partitions=(n_partitions_x, n_partitions_y),
                min_distance=self.grid_size // self.max_n_partitions   # self.min_distance
            )
            # self.n_random_images = choice(range(1, len(self.patch_info)))
            self.n_random_images = choice(range(0, len(self.patch_info)))
            self.random_patch_indices = sample(range(len(self.patch_info)), k=self.n_random_images)
        return self.n_random_images + 1  # + 1 for a background image

    def copy_paste(
            self,
            images: List[torch.Tensor],
            binary_masks: List[torch.Tensor],
            # interpolation: str,
            category_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        dict_data: Dict[str, torch.Tensor] = dict()
        assert len(images) == self.n_random_images + 1, f"{len(images)} != {self.n_random_images + 1}"
        background_image: torch.Tensor = images[0]
        background_mask: torch.Tensor = binary_masks[0]

        background_image = resize(
            background_image, size=self.grid_size, edge="shorter", interpolation="bilinear"
        )
        background_image, padding, offset = random_crop(background_image, crop_size=self.grid_size, fill=0)

        background_mask = resize(background_mask, size=self.grid_size, edge="shorter", interpolation="nearest")
        background_mask, _, _ = random_crop(
            background_mask, crop_size=self.grid_size, fill=0, padding=padding, offset=offset
        )

        images = images[1:]
        binary_masks = binary_masks[1:]

        if category_ids is not None:
            background_category_id = category_ids[0]
            category_ids = category_ids[1:]

            # make a grid for a semantic mask
            assert len(category_ids) == len(binary_masks), f"{len(category_ids)} != {len(binary_masks)}"
            background_semantic_mask: torch.Tensor = deepcopy(background_mask)

            # assign semantic label
            background_semantic_mask *= background_category_id

        n_instances = 1  # 1 for an object in the background image
        for image_index, patch_index in enumerate(self.random_patch_indices):
            image: torch.Tensor = images[image_index]
            binary_mask: torch.Tensor = binary_masks[image_index]
            assert image.shape[-2:] == binary_mask.shape[-2:], f"{image.shape[-2]} != {binary_masks.shape[-2:]}"

            assert 0 <= torch.min(binary_mask) <= 1, f"{binary_mask.min()}"
            assert 0 <= torch.max(binary_mask) <= 1, f"{binary_mask.max()}"

            # cut a bounding box of an object
            ymin, ymax, xmin, xmax = mask_to_bbox(binary_mask)
            if (ymin, ymax, xmin, xmax) == (-1, -1, -1, -1):
                continue

            image = image[:, ymin: ymax, xmin: xmax]
            binary_mask = binary_mask[ymin: ymax, xmin: xmax]  # (ymax - ymin) x (xmax - xmin)

            # sanity check
            h_image, w_image = image.shape[-2:]
            assert binary_mask.shape[-2:] == (h_image, w_image)

            patch_info: Dict[str, int] = self.patch_info[patch_index]

            top, left = patch_info["top"], patch_info["left"]
            h_patch, w_patch = patch_info["height"], patch_info["width"]
            try:
                if h_patch < w_patch:
                    if h_image < w_image:
                        image: torch.Tensor = resize(
                            image, size=h_patch, edge="shorter", max_size=w_patch, interpolation="bilinear"
                        )
                        binary_mask: torch.Tensor = resize(
                            binary_mask, size=h_patch, edge="shorter", max_size=w_patch, interpolation="nearest"
                        )
                    elif h_image > w_image:
                        image: torch.Tensor = resize(
                            image, size=h_patch, edge="longer", interpolation="bilinear"
                        )
                        binary_mask: torch.Tensor = resize(
                            binary_mask, size=h_patch, edge="longer", interpolation="nearest"
                        )
                    else:
                        image: torch.Tensor = resize(
                            image, size=h_patch, edge="both", interpolation="bilinear"
                        )
                        binary_mask: torch.Tensor = resize(
                            binary_mask, size=h_patch, edge="both", interpolation="nearest"
                        )
                elif h_patch > w_patch:
                    if h_image < w_image:
                        image: torch.Tensor = resize(
                            image, size=w_patch, edge="longer", interpolation="bilinear"
                        )
                        binary_mask: torch.Tensor = resize(
                            binary_mask, size=w_patch, edge="longer", interpolation="nearest"
                        )

                    elif h_image > w_image:
                        image: torch.Tensor = resize(
                            image, size=w_patch, edge="shorter", max_size=h_patch, interpolation="bilinear"
                        )
                        binary_mask: torch.Tensor = resize(
                            binary_mask, size=w_patch, edge="shorter", max_size=h_patch, interpolation="nearest"
                        )

                    else:
                        image: torch.Tensor = resize(
                            image, size=w_patch, edge="both", interpolation="bilinear"
                        )
                        binary_mask: torch.Tensor = resize(
                            binary_mask, size=w_patch, edge="both", interpolation="nearest"
                        )
                else:
                    # h_patch == w_patch
                    image: torch.Tensor = resize(
                        image, size=h_patch, edge="longer", interpolation="bilinear"
                    )
                    binary_mask: torch.Tensor = resize(
                        binary_mask, size=h_patch, edge="longer", interpolation="nearest"
                    )
            except RuntimeError:
                # in case the resized image has 0 width or 0 height due to an extreme aspect ratio
                continue

            n_instances += 1

            h_bbox, w_bbox = binary_mask.shape[-2:]
            if h_bbox == h_patch:
                assert w_patch >= w_bbox, f"{w_patch} should be larger than {w_bbox}"
                offset_left = randint(left, left + (w_patch - w_bbox))
                offset_top = top  # ymin
            elif w_bbox == w_patch:
                assert h_patch >= h_bbox, f"{h_patch} should be larger than {h_bbox}"
                offset_top = randint(top, top + (h_patch - h_bbox))
                offset_left = left  # xmin
            else:
                raise ValueError(f"check new image size ({h_bbox}, {w_bbox}) and patch size ({h_patch, w_patch})")

            binary_mask = binary_mask.to(torch.bool)  # need this type conversion for a proper indexing below

            try:
                background_image[:, offset_top: offset_top + h_bbox, offset_left: offset_left + w_bbox][:, binary_mask] = \
                    image[:, binary_mask]
            except IndexError:
                print(background_image.shape)
                print(background_image[:, offset_top: offset_top + h_bbox, offset_left: offset_left + w_bbox].shape)
                print(image.shape, binary_mask.shape, offset_top, h_bbox, offset_left, w_bbox)
                raise IndexError

            background_mask[offset_top: offset_top + h_bbox, offset_left: offset_left + w_bbox][binary_mask] = \
                binary_mask[binary_mask].to(torch.int64) + (n_instances - 1)

            if category_ids is not None:
                background_semantic_mask[offset_top: offset_top + h_bbox, offset_left: offset_left + w_bbox][binary_mask] = \
                    binary_mask[binary_mask].to(torch.int64) * category_ids[image_index]

        if (background_mask == 0).sum() > 0 and (background_mask > 0).sum() > 0:
            # exclude background regions
            background_image[:, background_mask == 0] = torch.mean(
                background_image[:, background_mask > 0], dim=-1, keepdim=True
            )

        dict_data["image"] = background_image
        dict_data["instance_mask"] = torch.stack(
            [(background_mask == instance_id) for instance_id in range(1, n_instances + 1)], dim=0
        )
        if category_ids is not None:
            dict_data["semantic_mask"] = background_semantic_mask

        return dict_data
