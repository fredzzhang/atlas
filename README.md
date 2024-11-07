
> [!NOTE]
> The repository is still being cleaned up. More documentation will be released soon.

# Task Vectors with Learned Anisotropic Scaling

This repository contains the official PyTorch implementation for the NeurIPS'24 paper
> Frederic Z. Zhang, Paul Albert, Cristian Rodriguez-Opazo, Anton van den Hengel, Ehsan Abbasnejad.
_Knowledge Composition using Task Vectors with Learned Anisotropic Scaling_.
In Advances in Neural Information Processing Systems (NeurIPS), 2024.

<a href="http://arxiv.org/abs/2407.02880">Preprint</a>

## Abstract
> ...<br/>This paper builds on properties of task vectors and aims to answer (1) whether components of task vectors, particularly parameter blocks, exhibit similar characteristics, and (2) how such blocks can be used to enhance knowledge composition and transfer. To this end, we introduce aTLAS, an algorithm that linearly combines parameter blocks with different learned coefficients, resulting in anisotropic scaling at the task vector level. We show that such linear combinations explicitly exploit the low intrinsic dimensionality of pre-trained models, with only a few coefficients being the learnable parameters. Furthermore, composition of parameter blocks enables modular learning that effectively leverages the already learned representations, thereby reducing the dependency on large amounts of data. We demonstrate the effectiveness of our method in task arithmetic, few-shot recognition and test-time adaptation, with supervised or unsupervised objectives. In particular, we show that (1) learned anisotropic scaling allows task vectors to be more disentangled, causing less interference in composition; (2) task vector composition excels with scarce or no labelled data and is less prone to domain shift, thus leading to better generalisability; (3) mixing the most informative parameter blocks across different task vectors prior to training can reduce the memory footprint and improve the flexibility of knowledge transfer. Moreover, we show the potential of aTLAS as a parameter-efficient fine-tuning method, particularly with less data, and demonstrate that it can be easily scaled up for higher performance.

<img src="./assets/teaser_a.png" height="300">&nbsp;&nbsp;<img src="./assets/teaser_b.png" height="300">

# Citation
If you find our work useful for your research, please consider citing us
```bibtex
@inproceedings{atlas_neurips_2024 ,
  title     = {Knowledge Composition using Task Vectors with Learned Anisotropic Scaling},
  author    = {Zhang, Frederic Z and Albert, Paul and Rodriguez-Opazo, Cristian and van den Hengel, Anton and Abbasnejad, Ehsan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```

## Prerequisites
1. Create a `conda` environment and install the dependencies.
```
conda env create -f environment.yml
```
2. Download and prepare the [datasets](./DATASETS.md).

## Reproducing experiment results

### 1. Task negation
TBA
### 2. Task addition
TBA
### 3. Few-shot adaptation
TBA
### 4. Test-time adaptation
TBA
### 5. Parameter-efficient fine-tuning
TBA

<!-- ## Task addition

## Task negation

## Few-shot recognition

[train_fewshot.sh](train_fewshot.sh) provides examples of training commands for the few-shot setting.
Training for few-shot generalization requires access to the trained task vector checkpoints.
One command with launch experiments over the 22 datasets.
Per-dataset results are logged into the `{exp_name}/{seed}/results.txt` file.

## Test-time adaptation

Test-time adaptation results using aTLAS and UFM can be reproduced by running
```sh
python src/learn_ufm.py --model=ViT-B-32 --blockwise --exp_name results/ViT-B-32_aTLAS/testime/ 
``` -->

## Acknowledgement

This repository is largely based on the code provided by [Ilharco et al. (2022)](https://github.com/mlfoundations/task_vectors) and [Ortiz-Jimenez et al. (2023)](https://github.com/gortizji/tangent_task_arithmetic).