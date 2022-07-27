import torch
import torch.nn as nn
import torch.nn.init as init

import math

def exp(x):
    return torch.exp(x) if torch.is_tensor(x) else math.exp(x)

def sqrt(x):
    return torch.sqrt(x) if torch.is_tensor(x) else math.sqrt(x)

# proposition 1
def snake_variance(alpha):
    num = 1 + exp(-8 * alpha ** 2) - 2 * exp(-4 * alpha ** 2)
    return 1 + num / (8 * alpha ** 2)

alpha_max = 0.56045
max_std = sqrt(snake_variance(alpha_max)) # 1.0971017221681962

def snake_gain(x):
    if torch.is_tensor(x):  # asssume x is alpha
        return sqrt(snake_variance(x))
    elif x == 'approx':
        return 1
    elif x == 'max':
        return max_std
    else:
        raise ValueError('undefined gain')

# initialization functions for network parameters preceding a Snake non-linearity
# pass alpha as 'kind' to use the exact variance
def snake_kaiming_uniform_(tensor, kind='approx', mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    gain = snake_gain(kind)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def snake_kaiming_normal_(tensor, kind='approx', mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    gain = snake_gain(kind)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)

class SnakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, correction=None):
        sinaxsq = (alpha * x).sin_().square_()
        alpha_coeff = torch.reciprocal(alpha)
        out = torch.addcmul(x, alpha_coeff, sinaxsq, out=sinaxsq)
        if correction is not None:
            out.mul_(torch.reciprocal(correction))
            if correction.requires_grad:
                ctx.save_for_backward(x, alpha, correction, out)
                return out
        ctx.save_for_backward(x, alpha, correction)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, correction, *_ = ctx.saved_tensors
        ax = alpha * x
        if alpha.requires_grad:
            sinaxsq = torch.sin(ax).square_()
        sin2ax = torch.sin_(ax.mul_(2))
        dyda, dydc = None, None
        if alpha.requires_grad:
            # 1/a * (x * sin(2ax) - 1/a * sin(ax)^2)
            alpha_coeff = torch.reciprocal(alpha)
            dyda = sinaxsq.mul_(-alpha_coeff).addcmul_(x, sin2ax).mul_(alpha_coeff).mul_(grad_output)
        # 1 + sin(2ax)
        dydx = sin2ax.add_(1).mul_(grad_output)
        if correction is not None:
            correction_coeff = torch.reciprocal(correction)
            dydx.mul_(correction_coeff)
            if alpha.requires_grad:
                dyda.mul_(correction_coeff)
            if correction.requires_grad:
                out, = _
                dydc = out.mul(-correction_coeff).mul_(grad_output)
        return dydx, dyda, dydc

class Snake(nn.Module):
    def __init__(self, num_channels, init=0.5, correction=None):
        super().__init__()
        if init == 'periodic':
            # "for tasks with expected periodicity, larger a, 
            # usually from 5 to 50 tend to work well"
            # => use a gamma distribution with mean 5 and a fat tail
            gamma = torch.distributions.Gamma(2, 1 / 10)
            self.alpha = nn.Parameter(gamma.sample((num_channels,)))
        else:  # assume init is a constant
            self.alpha = nn.Parameter(init * torch.ones(num_channels))
        self.correction = correction
    
    def forward(self, x):
        # reference: x + torch.sin(self.alpha * x) ** 2 / self.alpha
        if self.correction == 'std':
            correction = torch.sqrt(snake_variance(self.alpha))
        elif self.correction == 'max':
            correction = self.alpha.new_full((1,), max_std)
        else:
            correction = None
        dims = (Ellipsis,) + (None,) * (x.ndim - self.alpha.ndim - 1)
        alpha = self.alpha[dims]
        if correction is not None:
            correction = correction[dims]
        return SnakeFunction.apply(x, alpha, correction)
