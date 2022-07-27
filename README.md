# Fused Snake activation

Memory-efficient PyTorch implementation of the Snake activation function from [Neural Networks Fail to Learn Periodic Functions and How to Fix It (arXiv:2006.08195)](https://arxiv.org/abs/2006.08195) storing no intermediate activations - should use about the same memory and compute as other activation functions.

Snake has been shown to model periodic data well, particularly when it comes to generalizing beyond regions of data seen during training. See also: [BigVGAN: A Universal Neural Vocoder with Large-Scale Training](https://arxiv.org/abs/2206.04658)

I also implemented the variance corrections and Kaiming initializations mentioned in section 5 of the paper. These are untested and due to the vagueness of section 5 I am unsure about their correctness.

Fused Snake activation:
```
In [1]: S = Snake(48).cuda()
   ...: x = torch.randn(100, 48, 10 * 16000, device='cuda')
   ...: s = S(x)

In [2]: torch.cuda.max_memory_allocated() / 1024 / 1024
Out[2]: 5860.0009765625

In [3]: %timeit S(x); torch.cuda.synchronize()
90 ms ± 3.99 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

PyTorch Snake activation:
```
In [1]: x = torch.randn(100, 48, 10 * 16000, device='cuda')
   ...: alpha = torch.randn(48, 1, device='cuda')
   ...: s = x + torch.sin(alpha * x) ** 2 / alpha

In [2]: torch.cuda.max_memory_allocated() / 1024 / 1024
Out[2]: 8790.00048828125

In [3]: %timeit x + torch.sin(alpha * x) ** 2 / alpha; torch.cuda.synchronize()
110 ms ± 4.15 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
