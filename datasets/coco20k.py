import os.path
from typing import Dict, List, Tuple, Union
import json
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from datasets.base_dataset import BaseDataset


class COCO20KDataset(BaseDataset):
    def __init__(self, dir_dataset: str):
        """
        COCO20K dataset is composed of 19,817 images from the COCO2014 train set. The list of filenames can be found at
        https://github.com/valeoai/LOST/blob/master/datasets/coco_20k_filenames.txt
        """
        super(COCO20KDataset, self).__init__()
        self.dir_dataset: str = dir_dataset

        self.p_annotations: str = f"{dir_dataset}/annotations/instances_train2014_sel20k.json"
        if not os.path.exists(self.p_annotations):
            p_coco20k_filenames: str = f"{dir_dataset}/coco_20k_filenames.txt"
            p_all_annotations: str = f"{dir_dataset}/annotations/instances_train2014.json"
            assert os.path.exists(p_coco20k_filenames)
            assert os.path.exists(p_all_annotations)
            self.select_coco_20k(
                dir_dataset=dir_dataset,
                p_coco20k_filenames=p_coco20k_filenames,
                p_all_annotations_file=p_all_annotations
            )

        self.coco = COCO(self.p_annotations)
        self.image_ids = self.get_image_ids(remove_images_without_mask=False)

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # normalization with ImageNet mean, std
        self.name: str = "coco20k"
        self.n_categories = 81  # note that the background class (0) is included in this number.

        print(
            '\n\n'
            "Dataset summary\n"
            f"Dataset name: {self.name}\n"
            # f"# images (train, val): ({len(self.image_ids)})\n"
            f"# categories: {self.n_categories}\n"
        )

    # adapted from https://github.com/valeoai/LOST/blob/fcedbecb644f18358a660ce58c739cc6374feda8/datasets.py#L336
    @staticmethod
    def select_coco_20k(
            dir_dataset: str,
            p_coco20k_filenames: str,
            p_all_annotations_file: str
    ) -> None:
        p_annotations: str = f"{dir_dataset}/annotations/instances_train2014_sel20k.json"

        print('Building COCO 20k dataset.')
        # load all annotations
        train2014 = json.load(open(p_all_annotations_file, "r"))

        # load selected images
        with open(p_coco20k_filenames, "r") as f:
            sel_20k = f.readlines()
            sel_20k = [s.replace("\n", "") for s in sel_20k]
        im20k = [str(int(s.split("_")[-1].split(".")[0])) for s in sel_20k]

        new_anno = []
        new_images = []

        for i in tqdm(im20k):
            new_anno.extend(
                [a for a in train2014["annotations"] if a["image_id"] == int(i)]
            )
            new_images.extend([a for a in train2014["images"] if a["id"] == int(i)])

        train2014_20k = {}
        train2014_20k["images"] = new_images
        train2014_20k["annotations"] = new_anno
        train2014_20k["categories"] = train2014["categories"]

        json.dump(train2014_20k, open(p_annotations, "w"))
        print('Done.')

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

        # try:
        #     semantic_mask: torch.Tensor = torch.stack([
        #         (instance_mask == 1) * category_ids[instance_id] for instance_id, instance_mask in enumerate(instance_masks)
        #     ], dim=0).sum(dim=0)  # H x W
        # except TypeError:
        #     # TypeError: 'NoneType' object is not iterable
        #     # this error occurs when there is no ground truth instance masks for an image
        #     semantic_mask: torch.Tensor = torch.zeros(size=image_size, dtype=torch.int64)

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
