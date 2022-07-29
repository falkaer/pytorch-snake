import torch
import torch.nn as nn
import torch.nn.init as init

import math

def exp(x):
    return torch.exp(x) if torch.is_tensor(x) else math.exp(x)

def sqrt(x):
    return torch.sqrt(x) if torch.is_tensor(x) else math.sqrt(x)

def snake_variance(alpha):
    num = 1 + exp(-8 * alpha ** 2) - 2 * exp(-4 * alpha ** 2)
    return 1 + num / (8 * alpha ** 2)

def snake_second_moment(alpha):
    num = 3 + exp(-8 * alpha ** 2) - 4 * exp(-2 * alpha ** 2)
    return 1 + num / (8 * alpha ** 2)

alpha_max_var = 0.5604532115
max_std = sqrt(snake_variance(alpha_max_var))  # 1.0971017221...

alpha_max_second_moment = 0.65797
max_second_moment_sqrt = sqrt(snake_second_moment(alpha_max_second_moment))  # 1.1787158655

def snake_gain(x):
    if torch.is_tensor(x):  # assume x is alpha
        return 1 / sqrt(snake_second_moment(x))
    elif x == 'approx':
        return 1
    elif x == 'max':
        return 1 / max_second_moment_sqrt
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
        alpha_coeff = torch.reciprocal(alpha)
        out = torch.empty_like(x)
        torch.mul(x, alpha, out=out).sin_().square_().mul_(alpha_coeff).add_(x)
        if correction is not None:
            out.mul_(torch.reciprocal(correction))
        ctx.save_for_backward(x, alpha, correction, out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, correction, out = ctx.saved_tensors
        dyda, dydc = None, None
        # 1 + sin(2ax)
        dydx = torch.empty_like(grad_output)
        torch.mul(x, 2 * alpha, out=dydx).add_(1)
        if correction is not None:
            correction_coeff = torch.reciprocal(correction)
            dydx.mul_(correction_coeff)
            if ctx.needs_input_grad[2]:
                dydc = torch.empty_like(grad_output)
                torch.mul(out, -correction_coeff, out=dydc)
                dydc.mul_(grad_output)
        if ctx.needs_input_grad[1]:
            # 1/a * (x * dydx - out)
            alpha_coeff = torch.reciprocal(alpha)
            dyda = torch.empty_like(grad_output)
            torch.mul(x, dydx, out=dyda).sub_(out).mul_(alpha_coeff)
            dyda.mul_(grad_output)
        dydx.mul_(grad_output)
        return dydx, dyda, dydc

class Snake(nn.Module):
    def __init__(self, num_channels, init=0.5, correction=None):
        super().__init__()
        if init == 'periodic':
            # "for tasks with expected periodicity, larger a, 
            # usually from 5 to 50 tend to work well"
            # => use a gamma distribution with median ~5 and a heavy right tail
            gamma = torch.distributions.Gamma(concentration=1.5, rate=0.1)
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
