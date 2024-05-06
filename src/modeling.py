import open_clip
import torch
import torch.nn as nn

from src import utils
from functorch import jvp, make_functional_with_buffers

class ImageEncoder(nn.Module):
    def __init__(self, args, keep_lang=False, state_dict=None):
        super().__init__()

        #print(f"Loading {args.model} pre-trained weights.")
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

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

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
    def load_from_state_dict(cls, args, state_dict):
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            args.model, pretrained="openai", cache_dir=args.openclip_cachedir
        )
        self.model.load_from_state_dict(state_dict)


class ClassificationHead(nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = nn.Parameter(biases.clone())
        else:
            self.bias = nn.Parameter(torch.zeros_like(self.bias))
            

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


class ImageClassifier(nn.Module):
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

    def forward(self, inputs, return_features=False, **kwargs):
        features = self.image_encoder(inputs)        
        outputs = self.classification_head(features)
        if return_features:
            return outputs, features
        return outputs
    
        
    def __call__(self, inputs, *args, **kwargs):
        return self.forward(inputs, *args, **kwargs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = nn.ModuleList(classification_heads)
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


class LinearizedModel_(nn.Module):
    def __init__(self, model, task_vectors, args) -> None:
        """A wrapper class to enable compositions of task vectors"""
        super().__init__()

        self.params0 = model.params0
        self.func0 = model.func0
        self.buffers0 = model.buffers0
        self._model_name = model._model_name

        dparams = []
        for tv in task_vectors:
            dp = [tv.vector[k] for k in tv.vector if k.startswith('model.params.')]
            dparams.append(dp)

        self.dparams = dparams
        
        self.attn_l = model.attn_l
        self.n = model.n
        self.names = model.names
                
        self.attn = args.attn
        self.layerwise = args.layerwise
                
        self.tv_cpu = args.tv_cpu
        self.update_tvs(task_vectors)
        self.tv_cpu = args.tv_cpu
        
    def update_tvs(self, task_vectors):
        if hasattr(self, "dparams"):
            del self.dparams
            
        dparams = []
        for tv in task_vectors:
            dp = [tv.vector[k] for k in tv.vector if k.startswith('model.params.')]
            dparams.append(dp)
                
        self.dparams = dparams

        device = self.params0[0].device
        
        if hasattr(self, "coef"):
            del self.coef
        if hasattr(self, "coef1"):
            del self.coef1
        
        self.coef1 = []
        if self.attn:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors), len(self.params0)))
            self.coef1 = nn.Parameter(torch.zeros(len(task_vectors), len(self.params0), 3))
        elif self.layerwise:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors), len(self.params0)))
        else:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors)))

    def _apply(self, fn):

        new_self = super()._apply(fn=fn)
        new_self.buffers0 = tuple(x.to(new_self.coef) for x in new_self.buffers0)
        return new_self
            
    def tv_to_device(self):
        '''
        Moves Tvs to the coef device.
        Usefull to limit memory usage when selecting task vectors/cpu mode
        '''
        if not self.tv_cpu:
            self.dparams = [[dp.to(self.coef) for dp in dparam] for dparam in self.dparams]
            
        return self        
             
    def __call__(self, x, zero_shot=False) -> torch.Tensor:            
        if self.attn:
            dparams = [sum([p.to(c.device, non_blocking=True) * c[i] for p, c in zip(dp, self.coef)]) if i not in self.attn_l else sum([torch.cat((p[:len(p)//3].to(c.device, non_blocking=True) * c[i, 0], p[len(p)//3:2*len(p)//3].to(c.device, non_blocking=True) * c[i, 1], p[2*len(p)//3:len(p)].to(c.device, non_blocking=True) * c[i, 2]))  for p, c in zip(dp, self.coef1)]) for i, dp in enumerate(zip(*self.dparams))]
        elif self.layerwise:            
            dparams = [sum([p.to(c.device, non_blocking=True) * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p.to(c.device, non_blocking=True) * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]

        out, dp = jvp(
            lambda param: self.func0(param, self.buffers0, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp

        def __del_(self):
            # Manual delete of the reference to task vectors. To improve
            del self.dparams
            del self
        
class ImageEncoder_(nn.Module):
    def __init__(self, model, task_vectors, args) -> None:
        """A wrapper class to enable compositions of task vectors"""
        super().__init__()
        
        if args.lora:
            import minlora
            from functools import partial
            default_lora_config = {  # specify which layers to add lora to, by default only add to linear layers
                torch.nn.Linear: {
                    "weight": partial(minlora.LoRAParametrization.from_linear, rank=4),
                },
            }
            
            minlora.add_lora(model, lora_config=default_lora_config)
        
        func, params, self.buffer = make_functional_with_buffers(model)
        self.base_func = [func]
        # NOTE This is important to avoid the following error
        # NotImplementedError: Cannot copy out of meta tensor; no data!
        self.func = lambda p, b, x: self.base_func[0](p, b, x)
        
        self.params = nn.ParameterList(params)
        self.attn_l = []
        n = 0
        self.names = []
        self.buffer_index = [-1] #-1 for unimportant values such as attn mask and num_batches_tracked (equal to 0/-inf in task vectors)
        for i, (p, (name, _)) in enumerate(zip(self.params, model.named_parameters())):
            if ".visual" not in name:
                p.requires_grad = False
            elif args.tune_clip:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
            self.names.append(name)
            
            if 'attn.in_proj' in name:
                self.attn_l.append(i)
                n+=1
            if ".bn" in name or ".downsample" in name:
                if ".bias" in name:
                    self.buffer_index.append(i-1)
                    self.buffer_index.append(i)
                    self.buffer_index.append(-1)#num_batches_tracked
                    
        self.bnames = []
        for i, (name, _) in enumerate(model.named_buffers()):
            self.bnames.append(name)
            
        self.attn = args.attn
        self.layerwise = args.layerwise
                
        # Copy the attributes from the image encoder.
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = model.cache_dir

        self.tv_cpu = args.tv_cpu
        self.update_tvs(task_vectors)
        self.tv_cpu = args.tv_cpu
        
    def update_tvs(self, task_vectors):
        if hasattr(self, "dparams"):            
            del self.dparams
            
        dparams = []
        bufs = []
        for tv in task_vectors:
            dp = [tv.vector[k] if k in tv.vector.keys() else torch.tensor([0.]) for k in self.names]
            dparams.append(dp)
            bp = [tv.vector[k] if k in tv.vector.keys() else torch.tensor([0.]) for k in self.bnames]
            bufs.append(bp)

        self.bufs = bufs                
        self.dparams = dparams

        device = self.params[0].device
        
        if hasattr(self, "coef"):
            del self.coef
        if hasattr(self, "coef1"):
            del self.coef1
        
        self.coef1 = []
        if self.attn:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
            self.coef1 = nn.Parameter(torch.zeros(len(task_vectors), len(self.params), 3))
        elif self.layerwise:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
        else:
            self.coef = nn.Parameter(torch.zeros(len(task_vectors)))
        
    def train(self, mode):
        new_self = super().train(mode)
        for name, child in new_self.base_func[0].stateless_model.model.visual.named_modules():
            if isinstance(child, nn.BatchNorm2d):
                if mode:
                    child.train()
                else:
                    child.eval()
        return new_self

    def _apply(self, fn):

        new_self = super()._apply(fn=fn)
        new_self.buffer = tuple(x.to(new_self.coef) for x in new_self.buffer)
        return new_self
    
    def tv_to_device(self):
        '''
        Moves Tvs to the coef device.
        Usefull to limit memory usage when selecting task vectors/cpu mode
        '''
        if not self.tv_cpu:
            self.dparams = [[dp.to(self.coef) for dp in dparam] for dparam in self.dparams]
            self.bufs = [[bp.to(self.coef) for bp in bparam] for bparam in self.bufs]
            
        return self        
        
    def __call__(self, x, zero_shot=False) -> torch.Tensor:
        if zero_shot:
            return self.func(self.params, self.buffer, x)
        
        if self.attn:
            dparams = [sum([p.to(c.device, non_blocking=True) * c[i] for p, c in zip(dp, self.coef)]) if i not in self.attn_l else sum([torch.cat((p[:len(p)//3].to(c.device, non_blocking=True) * c[i, 0], p[len(p)//3:2*len(p)//3].to(c.device, non_blocking=True) * c[i, 1], p[2*len(p)//3:len(p)].to(c.device, non_blocking=True) * c[i, 2]))  for p, c in zip(dp, self.coef1)]) for i, dp in enumerate(zip(*self.dparams))]
        elif self.layerwise:            
            dparams = [sum([p.to(c.device, non_blocking=True) * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p.to(c.device, non_blocking=True) * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
            
        new_params = [dp + p for i, (dp, p) in enumerate(zip(dparams, self.params))]
        #c[self.buffer_index[i]].detach()
        buffers = [sum([b.to(c.device, non_blocking=True) * 0. for b, c in zip(bp, self.coef)]) for i, bp in enumerate(zip(*self.bufs))]
        buffers = [(bp + b) for (bp, b) in zip(buffers, self.buffer)]
        return self.func(new_params, buffers, x)

    def __del_(self):
        # Manual delete of the reference to task vectors. Causes a memory leak otherwise.
        del self.dparams
        del self.buffer
        del self.base_func
        del self

