# 知识蒸馏的开山之作
## 一、背景
&emsp;&emsp;我们都知道，大模型诸如 BERT，RoBERTa 难以上线，而从头开始训练一个小模型往往又达不到大模型的效果。如何利用大模型学习到的 *知识*，来让小模型获得更好的效果？  
&emsp;&emsp;好模型的目标不是拟合训练数据，而是学习如何泛化到新的数据。Hinton 在论文中提出方法很简单，就是让学生模型 $S$ 的预测 *分布*，来拟合教师模型 $T$ 的预测 *分布*。为什么不采用伪标签的思想，将 $T$ 的输出 label 作为 $S$ 的输入，而是采用拟合 softmax 输出（分类问题）的方法？直观来看，后者比前者具有这样一个优势：经过训练后的原模型，其 softmax 分布包含有一定的知识。  
> The relative probabilities of incorrect answers tell us a lot about how the cumbersome model tends to generalize. An image of a BMW, for example, may only have a very small chance of being mistaken for a garbage truck, but that mistake is still many times more probable than mistaking it for a carrot.

## 二、具体流程
&emsp;&emsp;接上文，我们的目标是让新模型与原模型的 softmax 输出的分布充分接近。直接这样做是有问题的：softmax 输出是一个接近 one-hot 的向量，其中一个值很大，其他的都很小。相较类似 one-hot 这样的硬性输出，我们更希望输出更 *软* 一些。  
&emsp;&emsp;为了描述简便，考虑一个 $k$ 分类的问题。设输入 $x\in \mathbb{R}$， 两个模型的 logits 分别是 $\boldsymbol{v}=T(x)\in\mathbb{R}^k,\boldsymbol{z}=S(x)\in\mathbb{R}^k$。论文中引入了一个温度超参来调节 soft 程度：  
$$
\boldsymbol{p}_i=\frac{e^{\boldsymbol{v}_i/T}}{\sum\limits_{j=1}^k{e^{\boldsymbol{v}_j/T}}}
$$
&emsp;&emsp;这样，原模型 $T$ 输出分布为 $\boldsymbol{p}$，新模型 $S$ 的输出分布为 $\boldsymbol{q}$，训练时，为了拉近彼此的距离，最小化二者的交叉熵：  
$$
\min{\mathcal{L}}=\min{-\sum\limits_{i=1}^k p_i\log q_i}
$$
&emsp;&emsp;我们把式子写开：  
$$
\frac{\partial{\mathcal{L}}}{\partial{z_i}}=\frac{\partial{\mathcal{L}}}{\partial{q_i}}\cdot\frac{\partial{q_i}}{\partial{z_i}}
$$
&emsp;&emsp;显然：  
$$
\frac{\partial{\mathcal{L}}}{\partial{\boldsymbol{q}}}=\begin{bmatrix}
-\frac{p_1}{q_1} \\
\vdots \\
-\frac{p_k}{q_k} \\
\end{bmatrix}_{k\times1}
$$
&emsp;&emsp;而对于 $\frac{\partial{q_i}}{\partial{z_i}}$，记 $A=\sum\limits_{i=1}^k{e^{\boldsymbol{v}_i/T}}$ 有：  
$$
\begin{align}
\frac{\partial q_i}{\partial z_j}&=\frac{\partial{\frac{e^{z_i/T}}{A}}}{\partial{z_j}}\\
&=\frac{1}{A^2}(A\frac{\partial e^{z_i/T}}{\partial z_j}-e^{z_i/T}\cdot\frac{1}{T}e^{z_j/T})\\
&=\frac{1}{A}\frac{\partial e^{z_i/T}}{\partial z_j}-\frac{1}{T}\frac{e^{z_i/T}}{A}\frac{e^{z_j/T}}{A}\\
&=\frac{1}{A}\frac{\partial e^{z_i/T}}{\partial z_j}-\frac{1}{T}q_iq_j\\
&=\begin{cases} 
\frac{1}{T}(q_i-q_iq_j),  & i=j \\
-\frac{1}{T}q_iq_j, & i\neq j
\end{cases}
\end{align}
$$
&emsp;&emsp;所以：  
$$
\frac{\partial{\boldsymbol{q}}}{\partial{\boldsymbol{z}}}=\frac{1}{T}\begin{bmatrix}
q_1-q_1^2      & -q_1q_2 & \cdots & -q_1q_k\\
-q_2q_1 & q_2-q_2^2 & \cdots & -q_2q_k\\
\vdots & \vdots& \ddots & \vdots \\
-q_kq_1 & -q_kq_2 & \cdots &q_k-q_k^2
\end{bmatrix}_{k\times k}
$$
&emsp;&emsp;代入式 3，有  
$$
\begin{align}
\frac{\partial{\mathcal{L}}}{\partial{\boldsymbol z}}&=\frac{1}{T}\begin{bmatrix}
q_1-q_1^2      & -q_1q_2 & \cdots & -q_1q_k\\
-q_2q_1 & q_2-q_2^2 & \cdots & -q_2q_k\\
\vdots & \vdots& \ddots & \vdots \\
-q_kq_1 & -q_kq_2 & \cdots &q_k-q_k^2
\end{bmatrix}\begin{bmatrix}
-\frac{p_1}{q_1} \\
\vdots \\
-\frac{p_k}{q_k} \\
\end{bmatrix}\\
&=\frac{1}{T}\begin{bmatrix}
-p_1+\sum_{i=1}^k p_iq_1\\
-p_2+\sum_{i=1}^k p_iq_2\\
\vdots\\
-p_k+\sum_{i=1}^k p_iq_k\\
\end{bmatrix}\\
&=\frac{1}{T}(\boldsymbol{q}-\boldsymbol{p})
\end{align}
$$
&emsp;&emsp;所以：  
$$
\begin{align}
\frac{\partial{\mathcal{L}}}{\partial{z_i}}&=\frac{1}{T}(q_i-p_i)\\
&=\frac{1}{T}(\frac{e^{\boldsymbol{z}_i/T}}{\sum\limits_{j=1}^k{e^{\boldsymbol{z}_j/T}}}-\frac{e^{\boldsymbol{v}_i/T}}{\sum\limits_{j=1}^k{e^{\boldsymbol{v}_j/T}}})
\end{align}
$$
&emsp;&emsp;若 $T$ 充分大，且 logits 分布的均值为0，即 $\sum_i z_i=\sum_i v_i =0$ 时，有：  
$$
\begin{align}
\frac{\partial{\mathcal{L}}}{\partial{z_i}}&\approx\frac{1}{T}(\frac{1+{z_i/T}}{\sum\limits_{j=1}^k(1+{z_j/T})}-\frac{1+{v_i/T}}{\sum\limits_{j=1}^k(1+{v_j/T})})\\
&=\frac{1}{T}(\frac{1+{z_i/T}}{k}-\frac{1+{v_i/T}}{k})\\
&=\frac{1}{kT^2}(z_i-v_i)
\end{align}
$$
&emsp;&emsp;而如果我们换一种更加直接的思路，直接拟合两个模型的 logits，有：  
$$
\frac{\partial(\frac{1}{2}\sum_{i-1}^k (z_i-v_i)^2)}{\partial z_i}=z_i-v_i
$$
&emsp;&emsp;比较式 18 与式 19，我们可以得到两个结论：  
* 如果 T 很大，且 logits 分布的均值为 0 时，优化概率交叉熵和 logits 的平方差是等价的；（没什么实际意义，事实上 T 不能取太大的值）
* 如果考虑多任务学习，损失函数同时包含蒸馏 loss 与常规监督训练 loss，则蒸馏交叉熵 loss 需要乘以 $T^2$ 以保证两个 loss 在同一量级。最终训练 loss 为：$\mathcal{L}=(1-\alpha)CE(y,\boldsymbol{q})+\alpha T^2CE(\boldsymbol{q},\boldsymbol{p})$。