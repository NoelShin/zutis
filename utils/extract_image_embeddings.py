import os
from typing import Dict, List, Optional
from math import sqrt
import pickle as pkl
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import clip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


@torch.no_grad()
def extract_image_embeddings(
        p_images: List[str],
        model_name: str = "RN50",
        fp: Optional[str] = None,
        device: torch.device = torch.device("cuda:0"),
        batch_size: int = 256,
        n_workers: int = 16,
) -> Dict[str, torch.FloatTensor]:
    assert model_name in ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],\
        ValueError(model_name)
    size = {
        "RN50": 224,
        "RN50x16": 384,
        "RN50x64": 448,
        "ViT-B/32": 224,
        "ViT-B/16": 224,
        "ViT-L/14": 224,
        "ViT-L/14@336px": 336
    }[model_name]

    print(f"Extracting features using {model_name}...")
    model, preprocess = clip.load(model_name, device=device)
    if "RN" in model_name:
        # only for resnet architectures
        pos_emb = model.visual.attnpool.positional_embedding
        cls_token, patch_tokens = pos_emb[0], pos_emb[1:]

        h_feat = w_feat = int(sqrt(len(patch_tokens)))
        h_feat_new = w_feat_new = int(size // 32)  # 32 is the total stride of CLIP resnet visual encoder

        # h_feat x w_feat x n_dims -> n_dims x h_feat x w_feat
        patch_tokens = patch_tokens.view(h_feat, w_feat, -1).permute(2, 0, 1)
        resized_patch_tokens = F.interpolate(
            patch_tokens[None],
            size=(h_feat_new, w_feat_new),
            mode="bicubic",
            align_corners=True
        )[0].view(-1, h_feat_new * w_feat_new).permute(1, 0)  # h_feat_new * w_feat_new x n_dims
        model.visual.attnpool.positional_embedding = nn.Parameter(
            torch.cat([cls_token[None], resized_patch_tokens], dim=0)
        )

    dataset = SimpleDataset(p_images=p_images, size=size)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=n_workers)

    filename_to_image_embedding: Dict[str, torch.FloatTensor] = dict()

    n_total_iters = len(dataloader)
    for num_iter, dict_data in enumerate(tqdm(dataloader)):
        image, p_images = dict_data["image"], dict_data["p_image"]
        image_embeddings: torch.FloatTensor = model.encode_image(image.to(device))  # b x 1024
        image_embeddings = image_embeddings / torch.linalg.norm(image_embeddings, ord=2, dim=1, keepdim=True)  # b x 1024
        image_embeddings = image_embeddings.cpu()

        for i, p_image in enumerate(p_images):
            # float16 -> float32
            filename_to_image_embedding[os.path.basename(p_image)] = image_embeddings[i].to(torch.float32)

        if num_iter % (n_total_iters // 20) == 0 and fp is not None:
            print(f"({(num_iter + 1) / n_total_iters:.1f}%) filename_to_image_embedding is saved at {fp}.")
            pkl.dump(filename_to_image_embedding, open(fp, "wb"))

    if fp is not None:
        pkl.dump(filename_to_image_embedding, open(fp, "wb"))
    return filename_to_image_embedding


# https://github.com/NoelShin/reco/blob/master/utils/extract_image_embeddings.py#L84
class SimpleDataset(Dataset):
    def __init__(
            self,
            p_images: List[str],
            size: int
    ):
        self.p_images: List[str] = p_images
        self.transforms = Compose([
            Resize(size, interpolation=BICUBIC),
            CenterCrop(size),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def __len__(self):
        return len(self.p_images)

    def __getitem__(self, ind):
        return {
            "image": self.transforms(Image.open(self.p_images[ind])),
            "p_image": self.p_images[ind]
        }