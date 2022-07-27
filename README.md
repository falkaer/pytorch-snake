# Fused Snake activation

Memory-efficient PyTorch implementation of the Snake activation function from [Neural Networks Fail to Learn Periodic Functions and How to Fix It (arXiv:2006.08195)](https://arxiv.org/abs/2006.08195) storing no intermediate activations - should use about the same memory and compute as other activation functions.

I also implemented the corrections and Kaiming initializations mentioned in section 5 of the paper, but they were very vague so I am unsure about correctness.