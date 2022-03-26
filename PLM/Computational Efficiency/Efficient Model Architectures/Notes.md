# 高性能 Transformer
## 一、自注意力线性复杂度的探索
### 1. 从原始公式开始
&emsp;&emsp;回顾经典 Transformer-Based 的预训练模型，用的最多的还是  scaled-dot self-attention:  
$$
Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax\left(\boldsymbol{Q}\boldsymbol{K}^{\top}\right)\boldsymbol{V}
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