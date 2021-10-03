# Transformer中的位置编码
## 前言
&emsp;&emsp;Transformer模型抛弃了RNN作为序列学习的基本模型。我们知道，循环神经网络本身就是一种顺序结构，天生就包含了词在序列中的位置信息。当抛弃循环神经网络结构，完全采用Attention取而代之，这些词序信息就会丢失，模型就没有办法知道每个词在句子中的相对和绝对的位置信息。因此，有必要把**词序信号**加到词向量上帮助模型学习这些信息，PE就是用来解决这种问题的方法。

---

## 想法
1.将 [0,1] 均分给每个字，其中 0 给第一个字，1 给最后一个字，也即$PE=\frac{pos}{sequencelength - 1}$。缺点很明显，即无法知道在一个特定区间范围内到底存在多少个单词。

2.线性分配，即$PE=pos$。这种方法缺点也很明显，PE会变得非常大以至于掩盖原有的Embedding信息。同时，如果给定一个测试句子，模型在训练时很有可能没有看到过任何一个这样的长度的样本句子，这会严重影响模型的泛化能力。

&emsp;&emsp;所以我们应该思考，一个好的PE应该具有哪些特征？

---

## 优秀的PE是怎样的？
* 首先，它应该能刻画位置信息！
* 每个位置的PE应该独一无二；
* 任意两个句子之间，相同间隔的两个位置PE的“距离”应该相同。也即给定间隔 $l$​​，存在$\mathbf{W},\mathbf{B}$，使得$\mathbf{W}\overrightarrow{PE}_{pos}+\mathbf{B}=\overrightarrow{PE}_{pos+l}$，且 $\mathbf{W},\mathbf{B}$ 与在序列中的位置 $pos$ 无关​​；
* 模型应该能毫不费力地泛化到更长的句子；
* 值的绝对值较小。

---

## Transformer中的PE
&emsp;&emsp;我们再来看论文中提出的PE。需要说明的是，在Transformer中PE是不训练的，在保证效果的前提下做到了简便，而BERT及之后的模型是训练的，即nn.Embedding。
&emsp;&emsp;论文中的公式：
$$
\begin{alignat}{2}
PE_{(pos,2i)}&=sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})\\
PE_{(pos,2i+1)}&=cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
\end{alignat}
$$

其中：pos指序列中的位置，定义域为[0,sequence_length);
&emsp;&emsp;&emsp;2i或2i+1指字Embedding中2i或2i+1维度的值，显然$2i\in \{0,2,\cdots,d_{model}-2\},2i+1 \in \{1,3,\cdots,d_{model}-1\}$​；
&emsp;&emsp;&emsp;$d_{models}$​​​ 指Embedding dimension，文中为512，若自定义则一般为一偶数。

&emsp;&emsp;如果原公式看不清的话，可以写开成如下的形式。对于一个batch中的一个句子(序列)，其PE为：
$$
\overrightarrow{PE}=[\overrightarrow{PE}_{(0)},\cdots,\overrightarrow{PE}_{(pos)},\cdots,\overrightarrow{PE}_{(L)}]^{T}
$$
其中：pos仍指序列中的位置，L指sequence length。
&emsp;&emsp;对于每一个句子，其位置编码 $\overrightarrow{PE}$​​ 由每个字的位置编码 $\overrightarrow{PE}_{(pos)}$​​ 组合而成，而 $\overrightarrow{PE}_{(pos)}$​​ 的维度为$d_{model}$​ (行向量)​，即 $\overrightarrow{PE}_{(pos)} \in \mathbb{R}^{1 \times d_{model}}$​ ，所以 $\overrightarrow{PE} \in \mathbb{R}^{L \times d_{model}}$​。
&emsp;&emsp;而对于 $\overrightarrow{PE}_{(pos)}$​，有：
$$
\begin{alignat}{2}
\overrightarrow{PE}_{(pos)}&=[\overrightarrow{PE}_{(pos)}^{(0)},\cdots,\overrightarrow{PE}_{(pos)}^{(d_{model}-1)}] \\
\overrightarrow{PE}_{(pos)}^{(m)}&=
\begin{cases}
sin(\omega_i \cdot pos),\ if\ m=2i \\
cos(\omega_i \cdot pos),\ if\ m=2i+1 
\end{cases} \\
where\quad \omega_i&=\frac{1}{10000^{\frac{2i}{d_{model}}}},\ i \in \{0,1,\cdots,\frac{d_{model}}{2}-1\}
\end{alignat}
$$
&emsp;&emsp;所以最后展开即为：
$$
\begin{alignat}{2}
\overrightarrow{PE}=[\overrightarrow{PE}_{(0)},\cdots,\overrightarrow{PE}_{(pos)},\cdots,\overrightarrow{PE}_{(L)}]^{T} =

\begin{bmatrix}
\overrightarrow{PE}_{(0)}^{(0)}      & \cdots &   \overrightarrow{PE}_{(0)}^{(d_{model}-1)}    \\
\vdots & \ddots & \vdots \\
\overrightarrow{PE}_{(pos)}^{(0)}      & \cdots &   \overrightarrow{PE}_{(pos)}^{(d_{model}-1)}    \\
\vdots & \ddots & \vdots \\
\overrightarrow{PE}_{(L)}^{(0)}      & \cdots &   \overrightarrow{PE}_{(L)}^{(d_{model}-1)}    \\
\end{bmatrix}_{L \times d_{model}}

\end{alignat}
$$

## Transformer的PE何以称为优秀？
&emsp;&emsp;接下来我们将验证论文中的公式是否满足我们提出的几点要求。
1.值独一无二且较小：考虑输入序列长度为50，embedding dimension为128，则各位置的PE为：(图源见参考文献[1])

<center><img src="1.jpg"  style="zoom:100%;" width="110%"/></center>
2.它能够刻画位置信息，相同间隔 $k$ 的两个位置PE的“距离”应该相同。
&emsp;&emsp;文中有这样的一段话：

> We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.
> 我们选择正弦曲线函数，因为我们假设它能让模型很容易地学习关注相对位置，因为对于任何固定的偏移量 $k$， $PE_{pos+k}$ 可以表示成 $PE_{pos}$ 的线性函数。

&emsp;&emsp;如果我们能证明 $PE_{pos+k}$ 与 $PE_{pos}$ 间存在**与 $pos$ 无关的线性关系**，则既可以说明PE能刻画位置信息，又能说明相同间隔 $k$ 的两个位置PE的“距离”相同。
&emsp;&emsp;先定性分析。考虑公式：
$$
\begin{alignat}{2}
sin(\alpha + \beta)=sin(\alpha)cos(\beta)+cos(\alpha)sin(\beta) \\
cos(\alpha+\beta)=cos(\alpha)cos(\beta)+sin(\alpha)sin(\beta)
\end{alignat}
$$
&emsp;&emsp;所以 $\overrightarrow{PE}_{(pos+k)}$​​​ 可以由 $\overrightarrow{PE}_{(pos)}$​​​​与 $\overrightarrow{PE}_{(k)}$​​​ 确定。但这样还是无法说明变换与 $pos$​​​ 无关。
&emsp;&emsp;下面我们将证明：**给定行向量 $\overrightarrow{PE}_{(pos)} \in \mathbb{R}^{1 \times d_{model}}$​​​ 与间隔 $k \in \mathbb{R}$​​​，存在与 $pos$​​ 无关的​变换 $\mathbf{T
}(k) \in \mathbb{R}^{d_{model} \times d_{model}}$​​​，使得 $\overrightarrow{PE}_{(pos)} \mathbf{T}(k) = \overrightarrow{PE}_{(pos+k)} $​​​。**
&emsp;&emsp;不妨假设：$\mathbf{T}(k)$​​​​ 的形式为：
$$
\begin{alignat}{2}
\mathbf{T}(k)=

\begin{bmatrix}
\mathbf{\Phi}_{1}(k) & \mathbf{0} & \cdots & \mathbf{0} \\
\mathbf{0} & \mathbf{\Phi}_{2}(k) & \cdots & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{\Phi}_{\frac{d_{mdoel}}{2}}(k)
\end{bmatrix}_{d_{model} \times d_{model}}

\end{alignat}
$$
其中：$\mathbf{0}$​​ 为 $2 \times 2$​​ 的全零矩阵；主对角线上的 $\mathbf{\Phi}_{n}(k),\ n\in \{1,2,\cdots,\frac{d_{model}}{2}\}$​​ ​​为变换矩阵，定义:
$$
\mathbf{\Phi}_{n}(k)=
\begin{bmatrix}
cos(k \cdot r_{n}) & -sin(k \cdot r_{n}) \\
sin(k \cdot r_{n}) & cos(k \cdot r_{n})
\end{bmatrix}_{2 \times 2}
$$
其中 $r_{n}$​为系数。
&emsp;&emsp;所以：
$$
\begin{alignat}{2}
\overrightarrow{PE}_{(pos)} \mathbf{T}(k) &=[sin(pos \cdot \omega_0),cos(pos \cdot \omega_0) \cdots,sin(pos \cdot \omega_{\frac{d_{model}}{2}-1}),cos(pos \cdot \omega_{\frac{d_{model}}{2}-1})]
\begin{bmatrix}
\mathbf{\Phi}_{1}(k) & \mathbf{0} & \cdots & \mathbf{0} \\
\mathbf{0} & \mathbf{\Phi}_{2}(k) & \cdots & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{\Phi}_{\frac{d_{mdoel}}{2}}(k)
\end{bmatrix}\\

&=\begin{bmatrix}
[sin(pos \cdot \omega_0),cos(pos \cdot \omega_0)]\mathbf{\Phi}_{1}(k) 
\ , \ [sin(pos \cdot \omega_1),cos(pos \cdot \omega_1)]\mathbf{\Phi}_{2}(k)\ , \ \cdots\ , \ [sin(pos \cdot \omega_{\frac{d_{model}}{2}-1}),cos(pos \cdot \omega_{\frac{d_{model}}{2}-1})]\mathbf{\Phi}_{\frac{d_{mdoel}}{2}}(k)
\end{bmatrix}_{1 \times d_{model}}
\end{alignat}
$$
&emsp;&emsp;考虑第n个元素：$[sin(pos \cdot \omega_{n-1}),cos(pos \cdot \omega_{n-1})]\mathbf{\Phi}_{n}(k)$​​，带入 $\mathbf{\Phi}_{n}(k)$​，得：
$$
\begin{alignat}{2}
[sin(pos \cdot \omega_{n-1}),cos(pos \cdot \omega_{n-1})]\mathbf{\Phi}_{n}(k)&=[sin(pos \cdot \omega_{n-1}),cos(pos \cdot \omega_{n-1})]\begin{bmatrix}
cos(k \cdot r_{n}) & -sin(k \cdot r_{n}) \\
sin(k \cdot r_{n}) & cos(k \cdot r_{n})
\end{bmatrix}\\
&=[sin(pos \cdot \omega_{n-1})cos(k \cdot r_{n})+cos(pos \cdot \omega_{n-1})sin(k \cdot r_{n})\ ,\ -sin(pos \cdot \omega_{n-1})sin(k \cdot r_{n})+cos(pos \cdot \omega_{n-1})cos(k \cdot r_{n})]
\end{alignat}
$$
&emsp;&emsp;带入公式(8)、(9)得：
$$
\begin{alignat}{2}
&[sin(pos \cdot \omega_{n-1}),cos(pos \cdot \omega_{n-1})]\mathbf{\Phi}_{n}(k)\\
&=[sin(pos \cdot \omega_{n-1}+k \cdot r_{n})\ , \ cos(pos \cdot \omega_{n-1}+k \cdot r_{n})] \\
&when\ r_{n}=\omega_{n-1}\\
&=[sin(\omega_{n-1}(pos+k))\ , \ cos(\omega_{n-1}(pos+k))]
\end{alignat}
$$
&emsp;&emsp;所以取 $r_{n}=\omega_{n-1}$ 时，
$$
\mathbf{\Phi}_{n}(k)=
\begin{bmatrix}
cos(k \cdot \omega_{n-1}) & -sin(k \cdot \omega_{n-1}) \\
sin(k \cdot \omega_{n-1}) & cos(k \cdot \omega_{n-1})
\end{bmatrix}_{2 \times 2}
$$
显然变换 $\mathbf{T}(k)$ 与 $pos$ 无关。而将式(19)的结果带入式(15)，有 $\overrightarrow{PE}_{(pos)} \mathbf{T}(k) = \overrightarrow{PE}_{(pos+k)} $ 得证。

---

## 一些个人理解
1.Transformer中PE的源码：
```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    
def __init__(self, d_model, dropout, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

def forward(self, x):
    x = x + Variable(self.pe[:, :x.size(1)],
                     requires_grad=False)
    return self.dropout(x)
```
2.为什么PE的结果是与字Embedding相加而不是拼接？
&emsp;&emsp;~~好像没有什么理论解释，纯粹是硬编..~~使参数更少，节省内存；某些情况下效果更好……参考[Why add positional embedding instead of concatenate?](https://github.com/tensorflow/tensor2tensor/issues/1591)

## 参考
1. [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) 强推！
2. [【Transformer系列2】Transformer结构位置编码的详细解析1（相对位置关系的推导证明与个人理解）](https://blog.csdn.net/qq_41554005/article/details/117387118?ops_request_misc=&request_id=&biz_id=102&utm_term=transformers%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-5-117387118.pc_search_es_clickV2&spm=1018.2226.3001.4187)
3. [Transformer升级之路：1、Sinusoidal位置编码追根溯源](https://spaces.ac.cn/archives/8231)
4. [Transformer 中的 Positional Encoding](https://wmathor.com/index.php/archives/1453/)

