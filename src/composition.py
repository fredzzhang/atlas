"""Models with learnable weights on task vectors 

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import torch

from torch import nn
from functorch import jvp, make_functional_with_buffers
    
class WeightedImageEncoder(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True) -> None:
        """A wrapper class to enable compositions of task vectors

        Parameter:
        ----------
        model: nn.Module
            CLIP image encoder model.
        task_vectors: List[NonLinearTaskVector]
            List of task vectors to learn coefficients for.
        blockwise: bool, default: True
            Learn a coefficient for each parameter block.
        """
        super().__init__()

        func, params, self.buffer = make_functional_with_buffers(model)
        # NOTE This is important to avoid the following error
        # NotImplementedError: Cannot copy out of meta tensor; no data!
        self.func = lambda p, b, x: func(p, b, x)
        self.params = torch.nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False

        # Copy the attributes from the image encoder.
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = model.cache_dir

        self.dparams = [[tv.vector[k] for k in tv.vector] for tv in task_vectors]
        self.blockwise = blockwise
        if blockwise:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
        else:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors),))

    def _apply(self, fn):
        """Override method to relocate buffer list

        NOTE: This function signature is for PyTorch 1.13.1.
        Newer verions have added another optional argument `recurse=True`.
        """
        new_self = super()._apply(fn=fn)
        new_self.buffer = (fn(x) for x in new_self.buffer)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def forward(self, x) -> torch.Tensor:
        if self.blockwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
        new_params = [dp + p for dp, p in zip(dparams, self.params)]
        return self.func(new_params, self.buffer, x)

class WeightedLinearizedModel(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True) -> None:
        """A wrapper class to enable compositions of task vectors for linearised models
        
        Parameters:
        -----------
        model: nn.Module
            Linearised model using first-order Taylor expansion.
        task_vectors: List[LinearizedTaskVector]
            List of task vectors to learn coefficients for.
        blockwise: bool, default: True
            Learn a coefficient for each parameter block.
        """
        super().__init__()

        self.params0 = model.params0
        self.func0 = model.func0
        self.buffers0 = model.buffers0
        self._model_name = model._model_name

        self.dparams = [[tv.vector[k] for k in tv.vector if k.startswith('model.params.')] for tv in task_vectors]
        self.blockwise = blockwise
        if blockwise:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params0)))
        else:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors),))

    def _apply(self, fn):
        """Override method to relocate buffer list

        NOTE: This function signature is for PyTorch 1.13.1.
        Newer verions have added another optional argument `recurse=True`.
        """
        new_self = super()._apply(fn=fn)
        new_self.buffers0 = (fn(x) for x in new_self.buffers0)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def forward(self, x) -> torch.Tensor:
        if self.blockwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
        out, dp = jvp(
            lambda param: self.func0(param, self.buffers0, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp