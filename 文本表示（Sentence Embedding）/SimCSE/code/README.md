# SimCSE 的复现与实验
&emsp;&emsp;主要参考 https://github.com/yangjianxin1/SimCSE 与 [官方代码](https://github.com/princeton-nlp/SimCSE)
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
| STS-B | 76.85 | 77.71 |

&emsp;&emsp;在 STS-B 上的调参过程：($\tau$=0.05，max_len=64)  
|  | LR | Batch Size | Dropout | 开发集上最佳指标 | 测试集上指标 |
| :---: | :---: |  :---: |  :---: |  :---: |  :---: |
| baseline | 3e-5 | 64 | 0.1 | 80.55 | 74.36 |
| 改dropout | 3e-5 | 64 | 0.15 | 80.97 | 77.71 |
| 改dropout | 3e-5 | 64 | 0.2 | 80.63 | 76.68 |
| 改dropout | 3e-5 | 64 | 0.3 | 78.27 | 73.81 |
| 改lr | 5e-5 | 64 | 0.1 | 80.70 | 76.23 |
| 改bs | 3e-5 | 128 | 0.1 | 79.07 | 73.73 |


## 二、有监督训练复现
### 1. 训练数据集
&emsp;&emsp;此处有监督训练并不是在 STS-B train 上完成的，而是使用 NLI 数据集构造。具体参见原论文。  

### 2. 复现指标
| | SimCSE 原论文 | 复现 |
| :---: |  :---: |  :---: |
| STS-B | 84.25 | 84.81 |

&emsp;&emsp;在 STS-B 上的调参过程：($mlm$=0.15，$\tau$=0.05，max_len=64，epochs=2)  
|  | $\lambda$ | Batch Size | Dropout | LR | dev | test |
| :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
| baseline | 0* | 128 | 0.1 | 5e-5 | 86.34 | 84.81 |
| 改bs | 0* | 64 | 0.1 | 5e-5 | 86.09 | 83.82 |
| 改dropout | 0* | 128 | 0.15 | 5e-5 | 86.28 | 84.35 |
| 改dropout | 0* | 128 | 0.2 | 5e-5 | 86.29 | 83.93 |

&emsp;&emsp;加入 MLM 训练：($\tau$=0.05，max_len=64)  
|  | $\lambda$ | Batch Size | Dropout | LR | dev | test |
| :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
| 对照组 | 0* | 32 | 0.1 | 2e-5 | 85.30 | 83.52 |
| baseline | 0.1 | 32 | 0.1 | 2e-5 | 85.57 | 84.28 |
| 加大$\lambda$ | 0.15 | 32 | 0.1 | 2e-5 | 85.53 | 84.35 |
| 加大$\lambda$ | 0.2 | 32 | 0.1 | 2e-5 | 85.52 | 83.93 |

* *：MLM=0，即只有监督任务，不同时训练 MLM 任务
* 损失函数为： $\mathcal{L}=\mathcal{L}_{CL}+\lambda \mathcal{L}_{MLM}$
* MLM 的 batch size 怎么下降了这么多？贫穷，是我的错，不是模型的错...可想而知原论文 512 的 bs 是有多壕了...

&emsp;&emsp;值得一提的是，上述复现的所有训练都是最 simple 的，连 weight decay 与 lr scheduler 都没使用。如果再加上一些 trick 好好调下参，说不定还能更高~

## 三、魔改与想法
