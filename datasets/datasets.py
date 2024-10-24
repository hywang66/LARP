import copy
import inspect

datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']

    dataset_params = inspect.signature(datasets[dataset_spec['name']]).parameters
    if 'kwargs' not in dataset_params:
        dataset_args = {k: v for k, v in dataset_args.items() if k in dataset_params}

    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset
