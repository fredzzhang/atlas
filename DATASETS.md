# Dataset preparation

All datasets are assumed to located under directory `~/data/`. The directory should be created if it does not already exist. To change the location of the datasets, override the default value of the argument `--data-location` in `atlas/src/args.py`. Most of the datasets will be downloaded automatically when running related training/inference scripts. However, several datasets require manual download and setup, and have been highlighted below.

* Cars - [manual setup required](#setup-guide-for-stanford-cars-dataset)
* DTD - [manual setup required](#setup-guide-for-dtd)
* EuroSAT - [manual setup required](setup-guide-for-eurosat)
* GTSRB - automatically downloaded
* MNIST - automatically downloaded
* RESISC45 - [manual setup required](#setup-guide-for-resisc45)
* SUN397 - [manual setup required](#setup-guide-for-sun397)
* CIFAR10 - automatically downloaded
* CIFAR100 - automatically downloaded
* ImageNet - [manual setup required](#setup-guide-for-imagenet)
* STL10 - automatically downloaded
* Food101 - automatically downloaded
* Caltech101 - automatically downloaded
* Caltech256 - automatically downloaded
* FGVCAircraft - automatically downloaded
* Flowers102 - automatically downloaded
* OxfordIIITPet - automatically downloaded
* CUB200 - automatically downloaded
* PascalVOC - automatically downloaded
* Country211 - automatically downloaded
* UCF101 - [manual setup required](#setup-guide-for-ucf)

## Setup Guide for Stanford Cars

As of 23 Nov. 2023, the download link from the project page [link](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) no longer works. As reported [here](https://github.com/pytorch/vision/issues/7545), the dataset homepage is offline. This [post](https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616) shows a quick guide to set up the dataset with the correct folder structure.

1. Download the dataset from [kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset?datasetId=30084&sortBy=dateCreated&select=cars_test).
2. Download the [devkit](https://github.com/pytorch/vision/files/11644847/car_devkit.tgz).
3. Download `cars_test_annos_withlabels.mat` file. [Here](https://github.com/nguyentruonglau/stanford-cars/blob/main/labeldata/cars_test_annos_withlabels.mat) is one possible source.

The directories should be set up in the following structure.

```
└── stanford_cars
    └── cars_test_annos_withlabels.mat
    └── cars_train
        └── *.jpg
    └── cars_test
        └── .*jpg
    └── devkit
        ├── cars_meta.mat
        ├── cars_test_annos.mat
        ├── cars_train_annos.mat
        ├── eval_train.m
        ├── README.txt
        └── train_perfect_preds.txt
```

## Setup Guide for DTD

The dataset has a working project [page](https://www.robots.ox.ac.uk/~vgg/data/dtd/), where is can be downloaded using this [link](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz). The dataset should be organised in the following structure.

```
└── dtd
    └──train
        └── banded
        └── braided
        └── ...
    └── test
        └── banded
        └── braided
        └── ...
    └── val
        └── banded
        └── braided
        └── ...
```

Out of the 10 balanced splits, the first one is used. The directory structure above can be achieved using the script below.

```python
import os
import shutil

def create_directory_structure(base_dir, classes, split='1'):
    target_dir = {'train': 'train', 'val': 'train', 'test': 'val'}
    for dataset in ['train', 'val', 'test']:
        path = os.path.join(base_dir, target_dir[dataset])
        os.makedirs(path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(path, cls), exist_ok=True)

        split_file = f'{dataset}{split}.txt'
        with open(os.path.join(base_dir, 'labels', split_file), 'r') as f:
            for line in f:
                line = line.strip()
                src_path = os.path.join(base_dir, 'images', line)
                dst_path = os.path.join(base_dir, target_dir[dataset], line)
                print(src_path, dst_path)
                shutil.copy(src_path, dst_path)

# Replace with the absolute path to your dataset
base_dir = '/path/to/data/dtd'            

classes = [d for d in os.listdir(os.path.join(base_dir, 'images')) if os.path.isdir(os.path.join(base_dir, 'images', d))]

create_directory_structure(base_dir, classes)
```

## Setup Guide for EuroSAT

The dataset has a Github [page](https://github.com/phelber/EuroSAT) with the necessary information. The dataset can be downloaded [here](https://madm.dfki.de/files/sentinel/EuroSAT.zip). After extraction, the dataset should have a folder named `2750`. The desired folder structure should be as follows

```
└── EuroSAT_splits
    └── train
        └── AnnualCrop
        └── River
        └── ...
    └── test
        └── AnnualCrop
        └── River
        └── ...
    └── val
        └── AnnualCrop
        └── River
        └── ...
```

This can be achieved using the script below.

```python
import os
import shutil
import random

def create_directory_structure(base_dir, classes):
    for dataset in ['train', 'val', 'test']:
        path = os.path.join(base_dir, dataset)
        os.makedirs(path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(path, cls), exist_ok=True)

def split_dataset(base_dir, source_dir, classes, val_size=270, test_size=270):
    for cls in classes:
        class_path = os.path.join(source_dir, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        val_images = images[:val_size]
        test_images = images[val_size:val_size + test_size]
        train_images = images[val_size + test_size:]

        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, 'train', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, 'val', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, 'test', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)

# Replace with the absolute path to your dataset
source_dir = '/path/to/data/2750'
base_dir = '/path/to/data/EuroSAT_splits'

classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

create_directory_structure(base_dir, classes)
split_dataset(base_dir, source_dir, classes)
```

## Setup Guide for RESISC45

The download links for the dataset as well as the splits are available in the dataset definition script. The relevant excerpt is shown below.

```python
url = "https://drive.google.com/file/d/1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv"
filename = "NWPU-RESISC45.rar"
split_urls = {
    "train": "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt",    # noqa: E501
    "val": "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt",        # noqa: E501
    "test": "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt",      # noqa: E501
}
```

The dataset also needs to be arranged in separate folders based on the splits, as well as the class names. This can be achieved by the script below.

```python
import os
import shutil

def create_directory_structure(data_root, split):
    split_file = f'resisc45-{split}.txt'
    with open(os.path.join(data_root, split_file), 'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip()
        class_name = '_'.join(l.split('_')[:-1])
        class_dir = os.path.join(data_root, 'NWPU-RESISC45', class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        src_path = os.path.join(data_root, 'NWPU-RESISC45', l)
        dst_path = os.path.join(class_dir, l)
        print(src_path, dst_path)
        shutil.move(src_path, dst_path)

data_root = '/home/frederic/data/resisc45'
for split in ['train', 'val', 'test']:
    create_directory_structure(data_root, split)
```

## Setup Guide for SUN397

The dataset has a working [project page](https://vision.princeton.edu/projects/2010/SUN/). Images and partitions of the dataset can be downloaded from said project page. After decompression, the dataset should have the following structure (note the lower case).

```
└── sun397
    └── split10.mat
    └── Training_01.txt
    └── Testing_01.txt
    └── ...
    └── a
    └── b
    └── ...
```

Out of the 10 different balanced splits, the first split is used. To set up the dataset with the desired folder structure, the images need to be put under dedicated `train` and `val` folders under the dataset root directory. Below is a script for this purpose.

```python
## PROCESS SUN397 DATASET

import os
import shutil
from pathlib import Path


def process_dataset(txt_file, downloaded_data_path, output_folder):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        input_path = line.strip()
        final_folder_name = "_".join(x for x in input_path.split('/')[:-1])[1:]
        filename = input_path.split('/')[-1]
        output_class_folder = os.path.join(output_folder, final_folder_name)

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        full_input_path = os.path.join(downloaded_data_path, input_path[1:])
        output_file_path = os.path.join(output_class_folder, filename)
        shutil.copy(full_input_path, output_file_path)
        if i % 100 == 0:
            print(f"Processed {i}/{len(lines)} images")

# Replace with the abosulte path to the dataset
downloaded_data_path = "/path/to/data/sun397"

process_dataset('Training_01.txt', downloaded_data_path, os.path.join(downloaded_data_path, "train"))
process_dataset('Testing_01.txt', downloaded_data_path, os.path.join(downloaded_data_path, "val"))

```
## Setup Guide for ImageNet

The dataset can no longer be downloaded from the project page. Alternatively, [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) hosts a copy of the dataset. It can be directly downloaded using the command below.

```bash
wget -O imagenet.zip 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/6799/4225553/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1701484051&Signature=gDBRItCHwgNv4NsV0H9%2FdygIFH27f2REcDEQLsu7sWcxf7vOYoKExI%2B4rqdsNcxAVq8A7RoM4g356aVYd5HmLgpPkoOQTdxtZUIBOFG9CJBJ0vdsqt2GfhdqgwSqaetiXVklxLdjgsEthBzIy8yK0zwCnXwjSnotE0Yif0PX6qfpIt4ibE6ZfSdMQ2n7zpxdu5ciCa8iwA8tByxQsYwn81MkM4kaBD1KWjipJyZdrayPSj2UH9lUzf3rr%2FO2LR6VzPbpKKgOt6DxM0WHexmvunMehks6Jm5oGyCkZ79vQU52Rjgow3beuCzkz%2FWE%2FXMcFLt8sbTPiIEcCn8ylkywtw%3D%3D&response-content-disposition=attachment%3B+filename%3Dimagenet-object-localization-challenge.zip'
```

After extracting the `.zip` file, it should have the following structure

```
└── ILSVRC
    └── Annotations
        └── ...
    └── Data
        └── CLS-LOC
            └── train
                └── n01440764
                    └── *.JPEG
                └── n01443537
                    └── *.JPEG
                └── ...
            └── val
                └── *.JPEG
            └── test
                └── *.JPEG
    └── ImageSets
        └── ...
```

As such, the validation set needs to be structured based on the classes, same as the training set. This [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) can be used for the purpose. In case the link becomes invalid in the future, a copy of the script is stored [locally](./valprep.sh).

## Setup Guide for UCF101

Download the extracted mid-frames [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing), and the dataset splits [here](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing). The `.zip` file should be extracted under `~/data/` and the `.json` file should be put under the dataset directory, with the following structure
```
└── UCF-101-midframes
    └── split_zhou_UCF101.json
    └── Apply_Eye_Makeup
        └── *.jpg
    └── ...
```