# Checkpoint preparation

The checkpoints are available for download on [HuggingFace](https://huggingface.co/fredzzhang/atlas/tree/main).
Downloaded files should be extracted and organised in the following structure.

```
└── checkpoints
    └── ViT-B-32
        └── zeroshot_accuracies.json
        └── ft_accuracies.json
        └── head_CarsVal.pt
        └── CarsVal
            └── zeroshot.pt
            └── finetuned.pt
        └── head_DTDVal.pt
        └── DTDVal
            └── zeroshot.pt
            └── finetuned.pt
        └── ...
    └── ViT-B-16
        └── ...
    └── ViT-L-14
        └── ...
    └── RN50
        └── ...
    └── RN101
        └── ...
```