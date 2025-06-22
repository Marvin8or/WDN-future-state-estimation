import torch


def adam_lr001(args, model_params):
    return args, torch.optim.Adam(params=model_params, lr=0.001)


def select_optimizer(args, model_params):
    assert args.optim_config in [
        "adam_lr001"
    ], "Optimizer configuration not present. Add optimizer configuration in 'optimizer_configurations.py'"
    if args.optim_config == "adam_lr001":
        args, optimizer = adam_lr001(args, model_params)
        return args, optimizer
