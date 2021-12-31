# Attention in Transformer
## 一、Scaled Dot-Product Attention
&emsp;&emsp;公式：
$$
Attention(\mathbf{Q},\mathbf{K},\mathbf{V})=softmax(\frac{\mathbf{Q}\mathbf{K}^T}{ \sqrt{d_k}})\mathbf{V}
$$
&emsp;&emsp;其中$\mathbf{Q}\in \mathbb{R}^{n\times d_k}$，$\mathbf{K}\in \mathbb{R}^{m\times d_k}$，$\mathbf{V}\in \mathbb{R}^{m\times d_v}$。而Self-Attention的情况下$\mathbf{Q}，\mathbf{K}，\mathbf{V}\in \mathbb{R}^{n\times d}，d=d_{model} / n\_heads$。本质是序列变换：从 $\mathbf{X}\in \mathbb{R}^{n \times d_{model}}$变换到$\mathbf{Z}\in \mathbb{R}^{n\times d}$。

$$
\mathbf{Z}=softmax(\frac{(\mathbf{XW_Q^T})(\mathbf{XW_K^T})^T}{\sqrt{d}})(\mathbf{XW_V^T})
$$

&emsp;&emsp;思考：
1. 为什么要除以 $\sqrt{d_k}$ ？

&emsp;&emsp;原文如下：
> We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

&emsp;&emsp;所以$\sqrt{d_k}$是一个调节因子，得内积不至于太大，否则softmax后就非0即1了，梯度会有问题。基于纯Transformer的实验如果不收敛，检查有没有除$\sqrt{d_k}$！！

2. 要不要softmax？

TODO

## 二、Multi-Head Attention
&emsp;&emsp;上文说到一次Self-Attention会把 $\mathbf{X}\in \mathbb{R}^{n \times d_{model}}$变换到$\mathbf{Z}\in \mathbb{R}^{n\times d}$，那么重复$n\_heads$次再拼接结果，得到Attention层的最终输出序列。
&emsp;&emsp;那么问题来了，重复$n\_heads$有何意义？原文是怎么说的：
> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

&emsp;&emsp;原论文中说到进行 Multi-head Attention 的原因是将模型分为多个头，形成多个子空间，可以让模型去关注不同方面的信息，最后再将各个方面的信息综合起来。其实直观上也可以想到，如果自己设计这样的一个模型，必然也不会只做一次 attention，多次 attention 综合的结果至少能够起到增强模型的作用，也可以类比 CNN 中同时使用多个卷积核的作用，直观上讲，多头的注意力有助于网络捕捉到更丰富的特征 / 信息。
