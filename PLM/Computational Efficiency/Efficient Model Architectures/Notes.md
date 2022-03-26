# 高性能 Transformer
## 一、自注意力线性复杂度的探索
### 1. 从原始公式开始
&emsp;&emsp;回顾经典 Transformer-Based 的预训练模型，用的最多的还是  scaled-dot self-attention:  
$$
Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax(\frac{\boldsymbol{Q}\boldsymbol{K}^{T}}{\sqrt{d_k}})\boldsymbol{V}
$$
&emsp;&emsp;其中 $\boldsymbol{Q}=\boldsymbol{K}=\boldsymbol{V}\in\mathbb{R}^{n\times d}$，为了简单考虑，忽略缩放因子 $d_k$。我们把公式1写开：  
$$
\begin{align}
Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_{ij} &=softmax((\boldsymbol{Q}\cdot\boldsymbol{K}
)_i)\cdot \boldsymbol{v}^T_j\\
&=\frac{\sum\limits_{r=1}^n e^{\boldsymbol{q}_i\cdot \boldsymbol{k}_r^T}\cdot v_{jr}^T}{\sum\limits_{r=1}^n e^{\boldsymbol{q}_i\cdot \boldsymbol{k}_r^T}}
\end{align}
$$
&emsp;&emsp;其中 $Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_{ij}$ 表示最终结果的第 $i$ 行 $j$ 列；$\boldsymbol{q}_i, \boldsymbol{k}_r,\boldsymbol{v}_j$ 均为 *行* 向量。  
&emsp;&emsp;同样有：  
$$
Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^n e^{\boldsymbol{q}_i\cdot\boldsymbol{k}_j^T} \cdot\boldsymbol{v}_j}{\sum\limits_{j=1}^n e^{\boldsymbol{q}_i\cdot\boldsymbol{k}_j^T}}
$$
&emsp;&emsp;所以本质上是以 $e^{\boldsymbol{q}_i\cdot\boldsymbol{k}_j^T}$ 对 $\boldsymbol{v}_j$ 做了加权平均，我们可以提出一个更广义上的 attention：  
$$
Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^n sim(\boldsymbol{q}_i, \boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n sim(\boldsymbol{q}_i, \boldsymbol{k}_j)}
$$
&emsp;&emsp;但是为了保留相似的分布特性，必须保证 $sim(\cdot,\cdot)\geq 0$。  

### 2. Kernel method
&emsp;&emsp;直接构造二元非负函数可能不太容易，我们可以转换另一个思路：如果对 $\boldsymbol{q}_i,\boldsymbol{k}_j$ 做一个非负变换，则他们的内积自然非负。我们有：  
$$
sim(\boldsymbol{q}_i, \boldsymbol{k}_j) = \phi(\boldsymbol{q}_i)\cdot\varphi(\boldsymbol{k}_j)^T
$$
&emsp;&emsp;特别地，如果我们取：$\phi = \varphi$，则我们可以将 $sim$ 视作两个 ***核函数*** 的内积（反向使用核方法）所以：  
$$
\begin{align}
Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_{i}&=
\frac{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)\cdot\varphi(\boldsymbol{k}_j)^T\cdot\boldsymbol{v}_j}{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)\cdot\varphi(\boldsymbol{k}_j)^T}\\
&=\frac{\phi(\boldsymbol{q}_i)\color{blue}{\sum\limits_{j=1}^n \varphi(\boldsymbol{k}_j)^T\cdot\boldsymbol{v}_j}}{\phi(\boldsymbol{q}_i)\color{blue}\sum\limits_{j=1}^n \varphi(\boldsymbol{k}_j)^T}
\end{align}
$$
&emsp;&emsp;这种思路的好处在于哪呢？？考虑到 $\boldsymbol{q}_i, \boldsymbol{k}_r,\boldsymbol{v}_j\in \mathbb{R}^d$，所以蓝色部分的复杂度为 $O(d^2)$，总体复杂度为 $O(n\cdot d^2)$，又 $d<<n$ （如在 BERT 中，由于 multi-head，d=64），所以时间复杂度可近似为 $O(n)$。同时由于原来需要存储中间结果矩阵 $\boldsymbol{Q}\boldsymbol{K}^{T}$ 用于计算 softmax，空间复杂度为 $O(n^2)$，现在只有矩阵乘法，空间复杂度也近似降为线性的了。所以拿掉 softmax 后就得到了我们梦想中的线性复杂度！  
&emsp;&emsp;我们现在还剩下一个问题：如何选择恰当的核函数？其中：[《Transformers are RNNs：Fast Autoregressive Transformers with Linear Attention》](https://github.com/HJHGJGHHG/NLPPapers/blob/main/PLM/Computational%20Efficiency/Efficient%20Model%20Architectures/Transformer%20to%20RNN/Transformers%20are%20RNNs%EF%BC%9AFast%20Autoregressive%20Transformers%20with%20Linear%20Attention.pdf) 给出的答案是：  
$$
\phi(x)=\varphi(x)=1 + elu(x) = \left\{\begin{aligned}1 + x,\, x \geq 0\\ e^x,\, x < 0\end{aligned}\right.
$$