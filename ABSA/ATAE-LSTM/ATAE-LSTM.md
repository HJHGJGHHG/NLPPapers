# AE, AT, ATAE-LSTM
&emsp;&emsp;基于论文：《Attention-based LSTM for Aspect-level Sentiment Classification》(EMNLP 2016) 。https://aclweb.org/anthology/D16-1058
## 一、Motivation
* 针对 TD, TC-LSTM 的出发点：Target-Dependent SA，提出 aspect 更为关键。（target 和 aspect 区别没有解释）原文：
> (About TD-LSTM and TC-LSTM)… However, those models can only take into consideration the target but not aspect information which is proved to be crucial for aspect-level classification.

* 指出 TC-LSTM 简单将目标词的各向量平均作为 target information 并不是最优的。
* Attention 机制在诸多领域取得了成功。

## 二、Models
### 1. LSTM with Aspect Embedding, AE-LSTM
&emsp;&emsp;为每个aspect，学习一个embedding向量，也就是学习一个 aspect embedding 矩阵。想法还是蛮朴素的，不好用已有的词向量表示 aspect，那就重新学习。

### 2.Attention-based LSTM, AT-LSTM
&emsp;&emsp;直接看模型图：
<center><img src="C:\Users\HJHGJGHHG\Desktop\论文笔记与复现\ABSA\Attention-based LSTM for Aspect-level Sentiment Classification\AT-LSTM.png"  style="zoom:30%;" width="100%"/></center>

&emsp;&emsp;模型首先通过一个LSTM模型得到每个词的隐藏状态向量，然后将其与Aspect Embedding连接，Aspect Embedding作为模型参数一起训练，从而得到句子在给定的aspect下的权值向量α，最后再根据权值向量对隐藏向量进行赋值，得到最终的句子表示，然后预测情感。

### 3.ATAE-LSTM
&emsp;&emsp;在 AT-LSTM 的基础上，在句子输入时额外再拼接对象词向量，就是 ATAE-LSTM 模型，即同时在模型的输入部分和隐态部分引入aspect信息。与 TC-LSTM的思想类似，使用这种方法进一步在句子表示中更好地利用目标词和每个上下文词之间的关系。模型结构如下图：
<center><img src="C:\Users\HJHGJGHHG\Desktop\论文笔记与复现\ABSA\Attention-based LSTM for Aspect-level Sentiment Classification\ATAE-LSTM.png"  style="zoom:30%;" width="100%"/></center>

## 三、Code
&emsp;&emsp;论文中的模型架构实现见 Models.py 中的 ATAE_LSTM，我自己魔改了一版，原论文相当于没有 Query，或者说 Q 为可训练的参数，我自己将 Aspect Embedding 作为 Query 得到 score，再与 LSTM output 相乘得到最后输出，不过效果不如原模型。魔改版见 ATAE_LSTM_Q。
