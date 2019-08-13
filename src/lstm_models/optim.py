from functools import partial
import math

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """Implements AdamW algorithm.
    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # according to the paper, this penalty should come after the bias correction
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)

        return loss


class ParamScheduler(object):
    
    def __init__(self, optimizer, step_size, start_lr=1e-4, max_lr=5e-3, last_lr=1e-3, warmup=0.2):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        
        self.optimizer = optimizer
        self.scale_fn = self.combine_scale_functions(
            [partial(self.scale_cos, start_lr, max_lr), partial(self.scale_cos, max_lr, last_lr)],
            [warmup, 1 - warmup])
        self.step_size = step_size
        self.last_batch_iteration = 0
        
    def batch_step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scale_fn(self.last_batch_iteration / self.step_size)
        
        self.last_batch_iteration += 1

    @staticmethod
    def combine_scale_functions(scale_fns, phases=None):
        if phases is None:
            phases = [1. / len(scale_fns)] * len(scale_fns)
        phases = [phase / sum(phases) for phase in phases]
        phases = torch.tensor([0] + phases)
        phases = torch.cumsum(phases, 0)
        
        def _inner(x):
            idx = (x >= phases).nonzero().max()
            actual_x = (x - phases[idx]) / (phases[idx + 1] - phases[idx])
            return scale_fns[idx](actual_x)
            
        return _inner

    @staticmethod
    def scale_cos(start, end, x):
        return start + (1 + np.cos(np.pi * (1 - x))) * (end - start) / 2
