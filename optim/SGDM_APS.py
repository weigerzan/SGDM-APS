import torch
import math
import copy

class SGDM_APS(torch.optim.Optimizer):
    def __init__(self, params, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False,
                 c=0.2,
                 gamma=2,
                 eps=0,
                 f_star=0,
                 warmup_steps=390
                 ):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        params = list(params)
        super(SGDM_APS, self).__init__(params, defaults)
        self.c=c
        self.gamma=gamma
        self.eps=eps
        self.momentum=momentum
        self.f_star = f_star
        self.params = params
        self.state['grad_last'] = None # sum(momentum^(k-m) * grad(fi(x)))
        self.state['diff'] = None # sum(fi(x)-fi*)*alpha^(1.5*(k-m))
        self.state['step_size'] = 100
        self.warmup_steps = warmup_steps
        self.d = 0.9999
    def __setstate__(self, state):
        super(PMSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None, loss=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # loss = None
        if loss == None:
            assert closure is not None
            with torch.enable_grad():
                loss = closure()
        current_grad = self.get_current_grad()
        if self.state['grad_last'] is None:
            grad_now = current_grad
        else:
            grad_now = self.state['grad_last'] # grad list
            for k in range(len(grad_now)):
                grad_now[k] = grad_now[k] * self.momentum + current_grad[k]
        # update grad
        self.state['grad_last'] = copy.deepcopy(grad_now)
        if self.state['diff'] is None:
            diff = loss - self.f_star
        else:            
            diff = self.state['diff'] * self.momentum ** 1.5 + loss - self.f_star
        self.state['diff'] = diff
        grad_norm = self.get_grad_norm()
        # print(grad_norm)
        if grad_norm > 1e-8:
            step = (1-math.sqrt(self.momentum)) * self.state['diff'] / (self.c * grad_norm ** 2 + self.eps)
            smooth_step = self.gamma ** (1/self.warmup_steps) * self.state['step_size']
            step_max = min(step.item(), smooth_step)
            
            self.state['step_size'] = step_max
            for p, g in zip(self.params, self.state['grad_last']):
                if isinstance(g, float) and g == 0.:
                    continue
                p.data.add_(other=g, alpha= -step_max)
        return loss

    def get_grad_norm(self):
        grad_norm = 0
        for g in self.state['grad_last']:
            if g is None:
                continue
            grad_norm += torch.sum(torch.mul(g, g))
        # print(grad_norm)
        return torch.sqrt(grad_norm)

    def get_current_grad(self):
        grad_list = []
        # print(self.params)
        for p in self.params:
            # print(p)
            if p.grad is None:
                g = 0
            else:
                g = p.grad.data
            grad_list.append(g)
        return grad_list