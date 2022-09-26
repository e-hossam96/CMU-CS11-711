from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]
                # initializing the state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros(grad.shape)
                    state['exp_avg_sq'] = torch.zeros(grad.shape)

                state['step'] += 1
                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                b1 = group['betas'][0]
                bt1 = b1 ** state['step']
                b2 = group['betas'][1]
                bt2 = b2 ** state['step']

                alphat = alpha * math.sqrt(1 - bt2) / (1 - bt1)
                m = state['exp_avg']
                v = state['exp_avg_sq']

                # Update first and second moments of the gradients
                m.mul_(b1).add_((1 - b1) * grad)
                v.mul_(b2).add_((1 - b2) * grad ** 2)

                # state['exp_avg'] = m
                # state['exp_avg_sq'] = v

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                # mhat = m / (1 - bt1)
                # vhat = v / (1 - bt2)

                # Update parameters
                p.data -= alphat * m / (torch.sqrt(v) + group['eps'])

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                p.data -= alpha * group['weight_decay'] * p.data


                # state["step"] += 1

                # step = state["step"]
                # exp_avg = state["exp_avg"]
                # exp_avg_sq = state["exp_avg_sq"]

                # # Access hyperparameters from the `group` dictionary
                # lr = group["lr"]
                # beta1, beta2 = group["betas"]
                # eps = group["eps"]
                # weight_decay = group["weight_decay"]

                # # Update first and second moments of the gradients
                # # In-place updates
                # exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                # exp_avg_sq.mul_(beta2).add_((1 - beta2) * grad * grad)

                # # Bias correction
                # # Please note that we are using the "efficient version" given in
                # # https://arxiv.org/abs/1412.6980
                # bias_corr1 = 1 - beta1 ** step
                # bias_corr2 = 1 - beta2 ** step
                # alpha = lr * math.sqrt(bias_corr2) / bias_corr1

                # # Update parameters
                # p.data.add_(-alpha * exp_avg / (exp_avg_sq.sqrt() + eps))

                # # Add weight decay after the main gradient-based updates.
                # # Please note that the learning rate should be incorporated into this update.
                # p.data.add_(-lr * weight_decay * p.data)

        return loss
