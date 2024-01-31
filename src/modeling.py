"""
Model definitions

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al.
at https://github.com/mlfoundations/task_vectors
"""

import open_clip
import torch

from src import utils

class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f"Loading {args.model} pre-trained weights.")
        if "__pretrained__" in args.model:
            name, pretrained = args.model.split("__pretrained__")
        elif "__init__" in args.model:
            print("Using random initialization.")
            name, pretrained = args.model.split("__init__")[0], None
        else:
            name = args.model
            pretrained = "openai"
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir
        )

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu")
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir
        )
        self.model.load_from_state_dict(state_dict)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)

class TaskVectorWithGrad:
    """A variant of the task vector with gradient enabled. It initialises the task vector
    from a pretrained checkpoints and a model's parameters that are being fine-tuned.
    
    NOTE: The definition of this class was included in this script as opposed to task_vectors.py,
    in order to avoid circular imports.
    """
    def __init__(self, pretrained_checkpoint, finetuned_model_params, device='cpu', linear=False):
        if linear:
            pretrained_state_dict = torch.load(pretrained_checkpoint)
            pretrained_state_dict.pop('model_name')
        else:
            pretrained_state_dict = torch.load(pretrained_checkpoint, map_location=device).state_dict()

        self.vector = {}
        for key, param in zip(pretrained_state_dict, finetuned_model_params):
            if pretrained_state_dict[key].dtype == torch.int64:
                continue
            if pretrained_state_dict[key].dtype == torch.uint8:
                continue
            self.vector[key] = (
                param - pretrained_state_dict[key]
            )

    def to(self, device):
        """Relocate the task vector to a designated device."""
        for k in self.vector:
            self.vector[k] = self.vector[k].to(device)

    def dot(self, other):
        """Dot product of two task vectors."""
        dot_product = 0.0
        for key in self.vector:
            if key not in other.vector:
                print(f"Warning, key {key} is not present in both task vectors.")
                continue
            dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))

    def sim(self, x):
        """Compute the cosine similarity against another task vector"""
        n = self.norm()
        if n == 0:
            return 0.0
        else:
            return torch.abs(self.dot(x) / (x.norm() * n))

class ImageClassifierWithOrthogReg(ImageClassifier):
    def __init__(self, pretrained_checkpoint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_checkpoint = pretrained_checkpoint

    def forward(self, task_vector, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        # Compute the orthognality regularisation term
        if task_vector is not None:
            params = self.image_encoder.parameters()
            w_delta = TaskVectorWithGrad(
                self.pretrained_checkpoint,
                params, device=outputs.device,
                linear='linear' in self.pretrained_checkpoint
            )
            reg = w_delta.sim(task_vector)
        else:
            reg = 0.0

        return outputs, reg
    
    def __call__(self, task_vector, *args, **kwargs):
        return self.forward(task_vector, *args, **kwargs)