# 0ESimCSE 复现
## 一、数据集
&emsp;&emsp;为了与 unsup-SimCSE 直接比较，此处还是使用 unsup-SimCSE 的 wiki 语料进行无监督训练。考虑到该数据集中有单个单词组成的句子，所以 Word Repetition 随机采样数修改为： $dup\_len \in [0,\max(1,int(dup\_rate\times N))]$

## 二、两个思路
### 1. 2N
&emsp;&emsp;复现的第一种思路即是借鉴自 SimCSE：将两个 view 合到一个 batch 中，这样得到 logits 仅需前向一次。输入时把两个 view concat 一起喂给 BERT，即：input 为：$logits=[(x_1,x_1^{’}),\cdots,(x_N,x_N^{’})]^T_{2N\times {sql\_len}}$，in-batch 的相似度矩阵为：
$$
\begin{bmatrix}
\frac{-1e^{12}}{\tau}      & \cdots & \frac{s_{1,2N}}{\tau}      \\
\vdots & \ddots & \vdots \\
\frac{s_{2N,1}}{\tau}      & \cdots & \frac{1e^{12}}{\tau}
\end{bmatrix}_{2N\times 2N}
$$
&emsp;&emsp;而 cross-batch 也就是 input 和 queue 中句嵌的相似度矩阵：
$$
\begin{bmatrix}
\frac{s_{1,Q_1}}{\tau}      & \cdots & \frac{s_{1,Q_{q\_len}}}{\tau}      \\
\vdots & \ddots & \vdots \\
\frac{s_{2N,Q_1}}{\tau}      & \cdots & \frac{s_{2N,Q_{q\_len}}}{\tau}
\end{bmatrix}_{2N\times Q}
$$
&emsp;&emsp;而后将二者 concat 作为最终的相似度矩阵：
$$
Sim=\left [
\begin{array}{c:c}
\begin{matrix}
\frac{-1e^{12}}{\tau}      & \cdots & \frac{s_{1,2N}}{\tau}      \\
\vdots & \ddots & \vdots \\
\frac{s_{2N,1}}{\tau}      & \cdots & \frac{-1e^{12}}{\tau}
\end{matrix}&
\begin{matrix}
\frac{s_{1,Q_1}}{\tau}      & \cdots & \frac{s_{1,Q_{q\_len}}}{\tau}      \\
\vdots & \ddots & \vdots \\
\frac{s_{2N,Q_1}}{\tau}      & \cdots & \frac{s_{2N,Q_{q\_len}}}{\tau}
\end{matrix}
\end{array}
\right ]_{2N\times (2N+q\_len)}
$$
&emsp;&emsp;而总 label 为：$[1,0,3,2,\cdots,2n-1,2n-2]_{2N\times 1}$，将二者做交叉熵得到 loss。
### 2. N

## 三、复现指标
### 1. 2N
&emsp;&emsp;调参如下：（dropout=0.1，Subword-Repetition，$\tau$=0.05）
|  | LR | Batch size | $\lambda$ | Queue_len | Dup_rate | 测试集 |
| :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
| baseline | 3e-5 | 64 | 0* | 0* | 0.2 | 77.61 |
| 改队列长度 | 3e-5 | 64 | 0.99 | 1 | 0.2 |  |
| 改队列长度 | 3e-5 | 64 | 0.99 | 1.5 | 0.2 |  |
| 改队列长度 | 3e-5 | 64 | 0.99 | 2 | 0.2 |  |
| 改队列长度 | 3e-5 | 64 |   0.99    | 2.5 | 0.2 |  |
| 改dup_rate | 3e-5 | 64 | 0* | 0* | 0.25 |  |
| 改dup_rate | 3e-5 | 64 | 0* | 0* | 0.32 |  |
| 改bs | 3e-5 | 96 | 0.99 | 2.5 | 0.32 |  |
| 改bs | 3e-5 | 128 | 0.99 | 2.5 | 0.32 |  |