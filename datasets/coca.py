from glob import glob
from typing import Dict, List
import numpy as np
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize


# Details of CoCA dataset can be found in http://zhaozhang.net/coca.html
class COCADataset(Dataset):
    def __init__(self, dir_dataset: str):
        super(COCADataset, self).__init__()
        self.dir_dataset: str = dir_dataset

        # noel: img path
        self.p_images: List[str] = sorted(glob(f"{dir_dataset}/image/**/*.jpg"))
        self.p_gts: List[str] = sorted(glob(f"{dir_dataset}/binary/**/*.png"))
        assert len(self.p_images) == len(self.p_gts)
        assert len(self.p_images) > 0

        self.n_categories: int = 1 + 80  # 1 for background
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.name: str = "coca"
        self.ignore_index: int = 255  # this ignore_index is taken into account for copy-paste aug. in an index dataset
        self.category_to_label_id: Dict[str, int] = {
            category: label_id for label_id, category in enumerate(coca_categories, start=1)
        }  # there is no fixed label ids for this dataset, so I simply made one based on a sorted list of the categories

    def __len__(self):
        return len(self.p_images)

    def __getitem__(self, index):
        p_image: str = self.p_images[index]
        p_gt: str = self.p_gts[index]
        category: str = p_gt.split('/')[-2]
        label_id: int = self.category_to_label_id[category]

        image = Image.open(p_image).convert("RGB")

        gt: torch.Tensor = torch.from_numpy(np.array(Image.open(p_gt))).to(torch.int64)  # {0, 255}
        gt[gt == 255] = label_id

        return {
            "image": normalize(to_tensor(image), mean=list(self.mean), std=list(self.std)),
            "p_image": p_image,
            "semantic_mask": gt,
            "p_semantic_mask": p_gt,
        }


# background category is not included and will be considered later in the code.
coca_categories: List[str] = [
    'Accordion', 'UAV', 'Yellow duck', 'alarm clock', 'avocado', 'backpack', 'baseball', 'beer bottle', 'belt',
    'binoculars', 'boots', 'butterfly', 'calculator', 'camel', 'camera', 'candle', 'chopsticks', 'clover', 'dice',
    'dolphin', 'doughnut', 'dumbbell', 'eggplant', 'faucet', 'fishing rod', 'frisbee', 'gift box', 'glasses', 'globe',
    'glove', 'guitar', 'hammer', 'hammock', 'handbag', 'harp', 'hat', 'headphone', 'helicopter', 'high heels',
    'hourglass', 'ice cream', 'key', 'lollipop', 'macaroon', 'microphone', 'minions', 'moon', 'persimmon', 'pigeon',
    'pillow', 'pine cone', 'pineapple', 'pocket watch', 'poker', 'potato', 'pumpkin', 'rabbit', 'rocking horse',
    'roller-skating', 'rolling pin', 'soap bubble', 'squirrel', 'stethoscope', 'sticky note', 'stool', 'strawberry',
    'sunflower', 'tablet', 'teddy bear', 'thermometer', 'tomato', 'towel', 'toy car', 'typewriter', 'violin', 'waffles',
    'watering can', 'watermelon', 'wheelchair', 'whisk'
]


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
coca_palette = create_pascal_label_colormap()  # 512 x 3

label_id_to_category: Dict[int, str] = {0: "background"}
label_id_to_category.update({label_id: category for label_id, category in enumerate(coca_categories, start=1)})
