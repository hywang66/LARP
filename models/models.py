import copy
import inspect
import torch

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False) -> torch.nn.Module:
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model_params = inspect.signature(models[model_spec['name']]).parameters
    if 'kwargs' not in model_params:
        model_args = {k: v for k, v in model_args.items() if k in model_params}
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'], strict=True)
    return model
