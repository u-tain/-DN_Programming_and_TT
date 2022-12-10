import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional

class AdaSmooth(Optimizer):
    def __init__(self, params, lr = 0.001, p1 = 0.5, p2 = 0.99, eps=1e-6, weight_decay=0, M = None ):
      '''
      Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        p1 (float, optional): fast decay constant (default: 0.5)
        p2 (float, optional): slow decay constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 0.001)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        M (int, optional): num of batchs per epoch (default: None)
      '''
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not p1 <= p2:
            raise ValueError("p2 must be > p1: p2 = {}, p1 = {}".format(p2, p1))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, 
                        p1=p1,
                        p2=p2, 
                        eps=eps, 
                        weight_decay=weight_decay,
                        M = M)
        super(AdaSmooth, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            norm_terms = []
            xt = []
            st = []
            nt = []
            lr, p1, p2, eps, weight_decay, M = (group['lr'],
                                            group['p1'],
                                            group['p2'],
                                            group['eps'],
                                            group['weight_decay'],
                                            group['M'])

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adasmooth does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['norm_terms'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['xt'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['st'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['nt'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                norm_terms.append(state['norm_terms'])
                xt.append(state['xt'])
                st.append(state['st'])
                nt.append(state['nt'])

                state['step'] += 1

            adasmooth(params_with_grad,
                     grads,
                     norm_terms,
                     xt,
                     st,
                     nt,
                     lr=lr,
                     p1=p1,
                     p2=p2,
                     eps=eps,
                     weight_decay=weight_decay,
                     M=M)

        return loss


def adasmooth(params: List[Tensor],
             grads: List[Tensor],
             norm_terms: List[Tensor],
             xt: List[Tensor],
             st: List[Tensor],
             nt: List[Tensor],
             *,
             lr: float,
             p1: float,
             p2: float,
             eps: float,
             weight_decay: float,
             M: float):
    step = 0
    for (param, grad, norm_term, x, s, n) in zip(params, grads, norm_terms, xt, st, nt):
        step +=1
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            norm_term = torch.view_as_real(norm_term)
            grad = torch.view_as_real(grad)
        s =  torch.add(param - x, s)
        n = torch.add(torch.abs(param - x), n)
        er = torch.div(torch.abs(s), n)
        c = torch.add(torch.mul((p2 - p1), er), (1 - p2)) #8
        norm_term = torch.add(torch.mul(c ** 2, torch.mul(grad, grad)), torch.mul((1 - c ** 2), norm_term)) #9
        delta = torch.mul( 1/ torch.sqrt(torch.add(norm_term,eps)),grad) # 10
        print()
        if torch.is_complex(param):
            delta = torch.view_as_complex(delta)
        param.add_(delta, alpha=-lr) # update
    if step == M:
        st = torch.zeros_like(p, memory_format=torch.preserve_format)
        nt = torch.zeros_like(p, memory_format=torch.preserve_format)