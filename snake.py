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
max_std = sqrt(snake_variance(alpha_max))  # 1.0971017221681962

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
        ctx.save_for_backward(x, alpha, correction, out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, correction, out = ctx.saved_tensors
        dyda, dydc = None, None
        # 1 + sin(2ax)
        dydx = torch.sin_((alpha * x).mul_(2)).add_(1)
        if correction is not None:
            correction_coeff = torch.reciprocal(correction)
            dydx.mul_(correction_coeff)
            if correction.requires_grad:
                dydc = out * -correction_coeff
                dydc.mul_(grad_output)
        if alpha.requires_grad:
            # 1/a * (x * dydx - out)
            alpha_coeff = torch.reciprocal(alpha)
            dyda = (x * dydx).sub_(out).mul_(alpha_coeff)
            dyda.mul_(grad_output)
        dydx.mul_(grad_output)
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

torch.manual_seed(0)
S = Snake(48).cuda()
S.alpha.requires_grad = False
x = torch.randn(100, 48, 10 * 16000, device='cuda')
s = S(x)
print(torch.cuda.max_memory_allocated() / 1024 / 1024)

x = torch.randn(100, 48, 10 * 16000, device='cuda')
alpha = torch.randn(48, 1, device='cuda')
s = x + torch.sin(alpha * x) ** 2 / alpha
print(torch.cuda.max_memory_allocated() / 1024 / 1024)
