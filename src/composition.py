"""Models with learnable weights on task vectors 

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import torch

from torch import nn

# from functorch import jvp, make_functional_with_buffers, make_functional

import copy


def make_functional(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values


def make_functional_with_buffers(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    print(params_names)
    params_values = tuple(params_dict.values())

    buffers_dict = dict(mod.named_buffers())
    buffers_names = buffers_dict.keys()
    buffers_values = tuple(buffers_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
        print(kwargs)
        print(args)
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        new_buffers_dict = {
            name: value for name, value in zip(buffers_names, new_buffers_values)
        }
        return torch.func.functional_call(
            stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs
        )

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values, buffers_values


class WeightedT5(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True) -> None:
        """A wrapper class to enable compositions of task vectors

        Parameter:
        ----------
        model: nn.Module
            T5 model.
        task_vectors: List[NonLinearTaskVector]
            List of task vectors to learn coefficients for.
        blockwise: bool, default: True
            Learn a coefficient for each parameter block.
        """
        super().__init__()

        func, params, self.buffer = make_functional_with_buffers(model)

        print(len(params), len(self.buffer))
        # NOTE This is important to avoid the following error
        # NotImplementedError: Cannot copy out of meta tensor; no data!
        self.func = lambda p, b, y, **x: func(p, b, y, **x)
        self.params = torch.nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False
        self.dparams = [[tv.vector[k] for k in tv.vector] for tv in task_vectors]
        self.blockwise = blockwise
        if blockwise:
            self.coef = torch.nn.Parameter(
                torch.zeros(len(task_vectors), len(self.params))
            )
        else:
            self.coef = torch.nn.Parameter(
                torch.zeros(
                    len(task_vectors),
                )
            )

    def _apply(self, fn):
        """Override method to relocate buffer list

        NOTE: This function signature is for PyTorch 1.13.1.
        Newer verions have added another optional argument `recurse=True`.
        """
        new_self = super()._apply(fn=fn)
        new_self.buffer = (fn(x) for x in new_self.buffer)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_training=False,
    ) -> torch.Tensor:

        if self.blockwise:
            dparams = []
            for i, dp in enumerate(zip(*self.dparams)):
                aux = []
                for p, c in zip(dp, self.coef):
                    aux.append(p * c[i])
                dparams.append(sum(aux))
            # dparams = [
            #     sum([p * c[i] for p, c in zip(dp, self.coef)])
            #     for i, dp in enumerate(zip(*self.dparams))
            # ]
        else:
            dparams = []
            for dp in zip(*self.dparams):
                aux = []
                for p, c in zip(dp, self.coef):
                    print(len(dp), c, p.shape)
                    aux.append(p * c)
                dparams.append(sum(aux))
            # dparams = [
            #     sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)
            # ]
        new_params = [dp + p for dp, p in zip(dparams, self.params)]
        # aux = [
        #     input_ids,
        #     attention_mask,
        #     decoder_input_ids,
        #     decoder_attention_mask,
        #     head_mask,
        #     cross_attn_head_mask,
        #     decoder_head_mask,
        #     encoder_outputs,
        #     past_key_values,
        #     inputs_embeds,
        #     decoder_inputs_embeds,
        #     labels,
        #     use_cache,
        #     output_attentions,
        #     output_hidden_states,
        #     return_dict,
        # ]
        aux = {
            # "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "decoder_head_mask": decoder_head_mask,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "decoder_inputs_embeds": decoder_inputs_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        return self.func(new_params, self.buffer, input_ids, **aux)


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
            self.coef = torch.nn.Parameter(
                torch.zeros(len(task_vectors), len(self.params))
            )
        else:
            self.coef = torch.nn.Parameter(
                torch.zeros(
                    len(task_vectors),
                )
            )

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
            dparams = [
                sum([p * c[i] for p, c in zip(dp, self.coef)])
                for i, dp in enumerate(zip(*self.dparams))
            ]
        else:
            dparams = [
                sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)
            ]
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

        self.dparams = [
            [tv.vector[k] for k in tv.vector if k.startswith("model.params.")]
            for tv in task_vectors
        ]
        self.blockwise = blockwise
        if blockwise:
            self.coef = torch.nn.Parameter(
                torch.zeros(len(task_vectors), len(self.params0))
            )
        else:
            self.coef = torch.nn.Parameter(
                torch.zeros(
                    len(task_vectors),
                )
            )

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
            dparams = [
                sum([p * c[i] for p, c in zip(dp, self.coef)])
                for i, dp in enumerate(zip(*self.dparams))
            ]
        else:
            dparams = [
                sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)
            ]
        out, dp = jvp(
            lambda param: self.func0(param, self.buffers0, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp
