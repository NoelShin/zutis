## Zero-shot Unsupervised Transfer Instance Segmentation
```
       ___           ___                                   ___     
      /  /\         /__/\          ___       ___          /  /\    
     /  /::|        \  \:\        /  /\     /  /\        /  /:/_   
    /  /:/:|         \  \:\      /  /:/    /  /:/       /  /:/ /\  
   /  /:/|:|__   ___  \  \:\    /  /:/    /__/::\      /  /:/ /::\
  /__/:/ |:| /\ /__/\  \__\:\  /  /::\    \__\/\:\__  /__/:/ /:/\:\
  \__\/  |:|/:/ \  \:\ /  /:/ /__/:/\:\      \  \:\/\ \  \:\/:/ /:/
      |  |:/:/   \  \:\  /:/  \__\/  \:\      \__\::/  \  \::/ /:/
      |  |::/     \  \:\/:/        \  \:\     /__/:/    \__\/ /:/  
      |  |:/       \  \::/          \__\/     \__\/       /__/:/   
      |__|/         \__\/                                 \__\/    


```    

Official PyTorch implementation of **Zero-shot Unsupervised Transfer Instance Segmentation**. Details can be found in the paper.
[[`paper`](https://arxiv.org/pdf/2304.14376.pdf)] [[`project page`](https://www.robots.ox.ac.uk/~vgg/research/zutis/)]

![Alt Text](project_page/assets/overview.png)

## Contents
* [Preparation](#preparation)
* [ZUTIS training/inference](#zutis-training/inference)
* [Pre-trained weights](#pre-trained-weights)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Preparation
### 1. Download datasets
Please download datasets of interest first by visiting the following links:

#### Instance segmentation
* [COCO-20K](https://cocodataset.org/#download)

Note that COCO20K dataset is composed of 19,817 images from the COCO2014 train set. The list of file names can be found at [this link](https://github.com/valeoai/LOST/blob/master/datasets/coco_20k_filenames.txt).

#### Semantic segmentation
* [CoCA](http://zhaozhang.net/coca.html)
* [COCO2017](https://cocodataset.org/#download)
* [ImageNet-S](https://github.com/LUSSeg/ImageNet-S)

Note that, we only consider object categories (and a background) for the COCO2017 dataset.

#### [Optional] Index datasets
If you want to construct image archives yourself for your custom dataset or a specific set of categories, you may want to download the following datasets to use as an index dataset (details can be found in [our paper](#)):
* [ImageNet2012](https://image-net.org/download.php)
* [PASS](https://www.robots.ox.ac.uk/~vgg/data/pass/)

We advise you to put the downloaded dataset(s) into the following directory structure for ease of implementation:
```bash
{your_dataset_directory}
├──coca
│  ├──binary
│  ├──image
├──coco  # for COCO-20K
│  ├──train2014
│  ├──annotations
│  ├──coco_20k_filenames.txt
├──coco2017
│  ├──annotations
│  ├──train2017
│  ├──val2017
├──ImageNet2012  # for an index dataset
│  ├──train
│  ├──val
├──ImageNet-S
│  ├──ImageNetS919
├──index_dataset
├──pass  # for an index dataset
│  ├──images
```

#### 2. Download required python packages:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib
conda install -c anaconda ujson
conda install -c conda-forge pyyaml
conda install -c conda-forge pycocotools
conda install -c anaconda scipy
pip install opencv-python
pip install git+https://github.com/openai/CLIP.git
```
A required version of each package might vary depending on your local device.

## ZUTIS training/inference

#### 0. Configuration file

Please change the following options to fit your directory structure before training/inference:
```yaml
dir_ckpt: {your_dir_ckpt}  # this should point to a checkpoint directory
dir_train_dataset:
[
    "{your_dataset_dir}/ImageNet2012/train",
    "{your_dataset_dir}/pass/images"
]  # this points to a directory of an index dataset(s)
p_filename_to_image_embedding: [
    "{your_dataset_dir}/ImageNet2012/filename_to_ViT_L_14_336px_train_img_embedding.pkl",
    "{your_dataset_dir}/pass/filename_to_ViT_L_14_336px_img_embedding.pkl"
]
dir_val_dataset: "{your_dataset_dir}/{evaluation_benchmark}",
category_to_p_images_fp: "{your_dataset_dir}/index_dataset/{evaluation_benchmark}_category_to_p_images_n500.json"
```

You can download `filename_to_image_embedding` and `category_to_p_images` files below.

**`filename_to_image_embedding`:**
- [`CLIP` image embeddings for the ImageNet2012 training set (~4.1GB)](#https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/imagenet2012/filename_to_ViT_L_14_336px_img_embedding.pkl)
- [`CLIP` image embeddings for PASS (~4.6GB)](#https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/pass/filename_to_ViT_L_14_336px_img_embedding.pkl)

Once downloaded, please put the files into the corresponding dataset directory (i.e., `ImageNet2012` and `pass` directories shown in the recommended directory structure above). Note that, in both cases, the `ViT-L/14@336px` `CLIP` image encoder is used to extract the image embeddings.

**`category_to_p_images`:**
- [CoCA (~3.2MB)](#https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/index_dataset/coca_category_to_p_images_n500.json)
- [COCO2017 (~3.2MB)](#https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/index_dataset/coco2017_category_to_p_images_n500.json)
- [ImageNetS919 (~37MB)](#https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/index_dataset/imagenet_s919_category_to_p_images_n500.json)

Please put these files in the `index_dataset` directory. In addition, you *have to* change the image paths in each file accordingly to your case.

#### 1. Training

`ZUTIS` is trained with pseudo-labels from an unsupervised saliency detector (e.g., `SelfMask`).
This involves two steps:
1. Retrieving images for a list of categories of interest from index datasets using `CLIP`;
2. Generating pseudo-masks for the retrieved images by applying `SelfMask` to them.

For the first step to be successfully done, make sure you already downloaded `CLIP` image embeddings for the images in the ImageNet2012 training set and in the PASS dataset as described in **0. Configuration file**.

The pseudo-mask generation process will be automatically triggered when running a training script, e.g., for training a model with a set of categories in COCO2017:

```shell
bash coco2017_vit_b_16.sh
```

It is worth noting that, as mentioned in the paper, the training is done for both semantic segmentation and instance segmentation at once. I.e., in the above case for COCO2017, the model will be trained with images for 80 object categories in COCO2017 to do semantic and instance segmentations.

#### 2. Inference
![Alt Text](project_page/assets/teaser.png)

**Semantic segmentation**

To evaluate a model with pre-trained weights on a semantic segmentation benchmark, e.g., COCO2017, please run:
```shell
bash coco2017_vit_b_16.sh $PATH_TO_WEIGHTS
```

**Instance segmentation**

For an instance segmentation benchmark, run:

```shell
bash coco20k_vit_b_16.sh $PATH_TO_WEIGHTS
```

## Pre-trained weights
We provide the pre-trained weights of ZUTIS:

#### Instance segmentation
benchmark| backbone | AP<sup>mk</sup> (%) | AP<sup>mk</sup><sub>50</sub> (%) |AP<sup>mk</sup><sub>75</sub> (%) |link|
:---:|:---:|:---:|:---:|:---:|:---:|
COCO-20K | ViT-B/16 | 5.7 | 11.0 | 5.4 |[weights](https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/coco2017/coco_vit_b_16.pt) (~537.9 MB)

#### Semantic segmentation
benchmark|split| backbone |IoU (%)|pixel accuracy (%)|link|
:---:|:---:|:---:|:---:|:---:|:---:|
CoCA | - | ViT-B/16 | 32.7 | 80.7 |[weights](https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/coca/coca_vit_b_16.pt) (~537.9 MB)
COCO2017 | val | ViT-B/16 | 32.8 | 76.4 |[weights](https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/coco2017/coco_vit_b_16.pt) (~537.9 MB)
ImageNet-S919 | test | ViT-B/32 | 27.5 | - |[weights](https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/imagenet-s919/imagenet_s919_vit_b_32.pt) (~544.6 MB)
ImageNet-S919 | test | ViT-B/16 | 37.4 | - |[weights](https://www.robots.ox.ac.uk/~vgg/research/zutis/shared_files/imagenet-s919/imagenet_s919_vit_b_16.pt) (~537.9 MB)

## Citation
```
@inproceedings{shin2023zutis,
  title = {Zero-shot Unsupervised Transfer Instance Segmentation},
  author = {Shin, Gyungin and Albanie, Samuel and Xie, Weidi},
  booktitle = {CVPRW},
  year = {2023}
}
```

## Acknowledgements
We borrowed code for `CLIP` from https://github.com/openai/CLIP.

If you have any questions about our code/implementation, please contact us at gyungin [at] robots [dot] ox [dot] ac [dot] uk.
