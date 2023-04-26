from typing import Dict, List, Tuple, Union
from glob import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from datasets.base_dataset import BaseDataset


class COCO2017Dataset(BaseDataset):
    def __init__(
            self,
            dir_dataset: str,
            split: str = "val",
    ):
        super(COCO2017Dataset, self).__init__()
        self.dir_dataset: str = dir_dataset
        self.split: str = split

        # val set
        self.p_annotations: str = f"{dir_dataset}/annotations/instances_val2017.json"
        self.coco = COCO(self.p_annotations)
        self.p_images: List[str] = sorted(glob(f"{dir_dataset}/val2017/*.jpg"))
        self.image_ids = self.get_image_ids(remove_images_without_mask=False)

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # normalization with ImageNet mean, std
        self.name: str = "coco2017"
        self.n_categories = 81  # note that the background class (0) is included in this number.

        print(
            '\n\n'
            "Dataset summary\n"
            f"Dataset name: {self.name}\n"
            # f"# images (train, val): ({len(self.image_ids)})\n"
            f"# categories: {self.n_categories}\n"
        )

    def get_image_ids(self, remove_images_without_mask: bool = True):
        image_ids: List[int] = self.coco.getImgIds()
        if remove_images_without_mask:
            list_new_image_ids: list = list()
            cnt = 0
            for image_id in tqdm(image_ids):
                ann_ids_per_img = self.coco.getAnnIds(imgIds=[image_id])
                anns_per_img: List[dict] = self.coco.loadAnns(ann_ids_per_img)
                if len(anns_per_img) > 0:
                    list_new_image_ids.append(image_id)
                else:
                    cnt += 1
            print(f"# filtered imgs (i.e., imgs w/o any mask): {cnt}")
            image_ids = list_new_image_ids
        return sorted(image_ids)

    def get_category_ids(self, image_id: int) -> List[int]:
        return self.coco.getCatIds(image_id)

    def get_image_path(self, image_id: int) -> str:
        img_info: Dict[str, Union[int, str]] = self.coco.loadImgs([image_id])[0]
        file_name = img_info["file_name"]

        # debug
        # img_path = f"{self.dir_dataset}/val2017/{file_name}"
        img_path = f"{self.dir_dataset}/{self.split}2017/{file_name}"
        return img_path

    def get_instance_masks_info(self, img_id: int) -> Dict[str, Union[torch.Tensor, List[int]]]:
        ann_ids_per_img = self.coco.getAnnIds(imgIds=[img_id])
        anns_per_img: List[dict] = self.coco.loadAnns(ann_ids_per_img)

        category_ids: List[int] = [ann["category_id"] for ann in anns_per_img]
        instance_masks: List[np.ndarray] = [self.coco.annToMask(ann) for ann in anns_per_img]
        assert len(category_ids) == len(instance_masks), f"{len(category_ids)} != {len(instance_masks)}"

        try:
            instance_masks: torch.Tensor = torch.from_numpy(np.stack(instance_masks, axis=0))  # M x h x w, torch.uint8
        except ValueError:
            # ValueError: need at least one array to stack
            instance_masks = None
        return {"instance_masks": instance_masks, "category_ids": category_ids}

    def __len__(self) -> int:
        return len(self.image_ids)

    @staticmethod
    def collate_fn(list_dict_data: List[dict]):
        # common
        batch_images: List[torch.Tensor] = list()
        batch_semantic_masks: List[torch.Tensor] = list()
        batch_instance_masks: List[List[torch.Tensor]] = list()  # [[mask 0, ..., mask N], ..., [mask 0', ..., mask N']]
        batch_category_ids: List[List[int]] = list()  # [[cat_id 0, ..., cat_id N], ..., [cat_id 0', ..., cat_id N']]
        batch_filenames: List[str] = list()
        batch_p_images: List[str] = list()

        # coco
        batch_image_ids: List[int] = list()
        batch_image_sizes: List[Tuple[int, int]] = list()

        for dict_data in list_dict_data:
            batch_images.append(dict_data["image"])
            batch_semantic_masks.append(dict_data["semantic_mask"])
            batch_instance_masks.append(dict_data["instance_mask"])
            batch_category_ids.append(dict_data["category_ids"])
            batch_filenames.append(dict_data["filename"])
            batch_p_images.append(dict_data["p_image"])

            batch_image_ids.append(dict_data["image_id"])
            batch_image_sizes.append(dict_data["image_size"])

        return {
            "image": torch.stack(batch_images, dim=0),
            "semantic_mask": torch.stack(batch_semantic_masks, dim=0),
            "instance_mask": batch_instance_masks,
            "category_ids": batch_category_ids,
            "filename": batch_filenames,
            "p_image": batch_p_images,
            "image_id": batch_image_ids,
            "image_size": batch_image_sizes
        }

    def __getitem__(self, index: int) -> dict:
        """Return a dictionary of data. If train mode, do data augmentation."""
        data: dict = dict()
        image_id: int = self.image_ids[index]
        p_image: str = self.get_image_path(image_id)
        image = Image.open(p_image).convert("RGB")
        image_size = [image.size[i] for i in (1, 0)]  # w x h -> h x w

        dict_instance_masks: Dict[str, Union[torch.Tensor, List[int]]] = self.get_instance_masks_info(image_id)
        instance_masks: torch.Tensor = dict_instance_masks["instance_masks"]  # n_instances x H x W
        category_ids: List[int] = dict_instance_masks["category_ids"]  # n_instances

        filename = p_image.split('/')[-1].split('.jpg')[0]
        p_semantic_mask: str = f"{self.dir_dataset}/annotations/semantic_segmentation_masks/{filename}.png"
        semantic_mask: torch.Tensor = torch.from_numpy(np.array(Image.open(p_semantic_mask))).to(torch.int64)  # H x W

        data.update({
            "image": TF.normalize(TF.to_tensor(image), self.mean, self.std),
            "semantic_mask": semantic_mask,
            "instance_mask": instance_masks,
            "category_ids": category_ids,

            "filename": p_image.split('/')[-1].split('.')[0],
            "p_image": p_image,

            "image_id": image_id,
            "image_size": image_size,
        })
        return data


old_label_id_to_new_label_id = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    # 12: "street sign", removed from COCO
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25: 24,
    # 26: "hat", removed from COCO
    27: 25,
    28: 26,
    # 29: "shoe", removed from COCO
    # 30: "eye glasses", removed from COCO
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    44: 40,
    # 45: "plate", removed from COCO
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    65: 60,
    # 66: "mirror", removed from COCO
    67: 61,
    # 68: "window", removed from COCO
    # 69: "desk", removed from COCO
    70: 62,
    # 71: "door", removed from COCO
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    81: 72,
    82: 73,
    # 83: "blender", removed from COCO
    84: 74,
    85: 75,
    86: 76,
    87: 77,
    88: 78,
    89: 79,
    90: 80,
}
new_label_id_to_old_label_id = {v: k for k, v in old_label_id_to_new_label_id.items()}


# copy-pasted from https://github.com/NoelShin/reco/blob/master/datasets/coco_stuff.py
def create_pascal_label_colormap() -> np.ndarray:
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    def bit_get(val, idx):
        """Gets the bit value.
        Args:
          val: Input value, int or numpy int array.
          idx: Which bit of the input val.
        Returns:
          The "idx"-th bit of input val.
        """
        return (val >> idx) & 1

    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap

coco2017_palette = create_pascal_label_colormap()  # 512 x 3
coco2017_palette[255] = np.array([255, 255, 255])  # for an ignore index (=255), white colour is assigned

label_id_to_category = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    # 12: "street sign", removed from COCO
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    # 26: "hat", removed from COCO
    27: "backpack",
    28: "umbrella",
    # 29: "shoe", removed from COCO
    # 30: "eye glasses", removed from COCO
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    # 45: "plate", removed from COCO
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    # 66: "mirror", removed from COCO
    67: "dining table",
    # 68: "window", removed from COCO
    # 69: "desk", removed from COCO
    70: "toilet",
    # 71: "door", removed from COCO
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    # 83: "blender", removed from COCO
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

label_id_to_rgb = {
    label_id: rgb
    for label_id, rgb in zip(label_id_to_category.keys(), coco2017_palette)
}
