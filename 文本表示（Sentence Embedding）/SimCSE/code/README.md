# SimCSE 的复现与实验
## 一、无监督训练复现
### 1. 训练数据集
&emsp;&emsp;如论文中所述，Unsupervised SimCSE 并不使用 STS 的训练集，而是使用从 Wikipedia 上爬取的一百万条句子做无监督训练。数据集下载：  
```
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
```

### 2. Label
&emsp;&emsp;虽然理论上我们说 SimCSE 是将同一个批次内的句子分别送入 PLM 两次，但实现的时候我们其实是将一个 batch 内的所有样本复制一遍，然后通过一次 Encoder 即可。假设初始输入为 $[A, B]$ 两条句子，首先复制一遍 $[A, A, B, B]$，那么经过 Encoder 得到的句向量为 $[\boldsymbol {h}_A^{(0)}, \boldsymbol {h}_A^{(1)}, \boldsymbol {h}_B^{(0)}, \boldsymbol {h}_B^{(1)}]$。  
&emsp;&emsp;其 label 如下：  
$$
\begin{array}{c|c|c|c|c} 
\hline 
& \boldsymbol{h}_A^{(0)} & \boldsymbol{h}_A^{(1)} & \boldsymbol{h}_B^{(0)} & \boldsymbol{h}_B^{(1)}\\ 
\hline 
\boldsymbol{h}_A^{(0)} & 0 & 1 & 0 & 0\\ 
\hline 
\boldsymbol{h}_A^{(1)} & 1 & 0 & 0 & 0\\ 
\hline 
\boldsymbol{h}_B^{(0)} & 0 & 0 & 0 & 1\\ 
\hline 
\boldsymbol{h}_B^{(1)} & 0 & 0 & 1 & 0\\ 
\hline 
\end{array}
$$
&emsp;&emsp;然后把 One-Hot 形式的 label 转换成 index，即 label=$[1,0,3,2]$。  

### 3. 复现指标
&emsp;&emsp;预训练模型取 bert-base-uncased，指标为 Spearman’s correlation。最佳指标如下（测试集）：  
| | SimCSE 原论文 | 复现 |
| :---: |  :---: |  :---: |
| STS-B | 76.85 |  |

&emsp;&emsp;在 STS-B 上的调参过程：($\tau$=0.05，max_len=64)  
|  | LR | Batch Size | Dropout | 开发集上最佳指标 | 测试集上指标 |
| :---: | :---: |  :---: |  :---: |  :---: |  :---: |
| baseline | 3e-5 | 64 | 0.1 | 80.55 | 74.36 |
| 改dropout | 3e-5 | 64 | 0.15 | | |
| 改dropout | 3e-5 | 64 | 0.2 | 80.63 | 76.68 |
| 改dropout | 3e-5 | 64 | 0.3 | | |
| 改lr | 5e-5 | 64 | 0.2 | | |
| 改bs | 3e-5 | 128 | 0.1 | | |
