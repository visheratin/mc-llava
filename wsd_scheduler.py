import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass

@dataclass
class WSDParameters:
    total_steps: int
    warmup_ratio: float
    stable_ratio: float
    annealing_ratio: float
    min_lr_ratio: float

def _wsd_scheduler_lambda(
    current_step: int, *, params: WSDParameters, num_cycles: float
):
    num_warmup_steps = int(params.total_steps * params.warmup_ratio)
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    num_stable_steps = int(params.total_steps * params.stable_ratio)
    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    num_annealing_steps = int(params.total_steps * params.annealing_ratio)
    if current_step < num_warmup_steps + num_stable_steps + num_annealing_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_annealing_steps))
        value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return (1.0 - params.min_lr_ratio) * value + params.min_lr_ratio
    return params.min_lr_ratio

def wsd_scheduler(
    optimizer: Optimizer, params: WSDParameters, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _wsd_scheduler_lambda,
        params=params,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)