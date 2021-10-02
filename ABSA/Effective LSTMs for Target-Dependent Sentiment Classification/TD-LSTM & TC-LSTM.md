# TD-LSTM & TC-LSTM
&emsp;&emsp;基于论文：《Effective LSTMs for Target-Dependent Sentiment Classification》2016。https://arxiv.org/abs/1512.01100
## 一、Target-dependent Sentiment Analysis, TDSA
&emsp;&emsp;给定目标词的情感分析：与典型情感分析任务不同，target-dependent 情感分析是研究基于目标的情感。给定一个句子和句子相关的一个对象，判断句子针对给定的对象的情感倾向。
&emsp;&emsp;例如，有句子：“**张三**在学校里**很受大家欢迎**，但是邻居***李四不太受欢迎*** ！”
&emsp;&emsp;其中，基于目标“张三”，句子的情感是正向的；基于“李四”，句子的情感是负面的。可见，与传统的情感分析任务相比，任务的难度和复杂性大大增加，一般都是用深度学习模型来解决。而在2016年，在 Transomer 与大规模预训练模型之前，主要采用 RNN、LSTM，也有 ML 模型(如 SVM)+手动特征工程。（论文 Introduction 部分）

---

## 二、TD-LSTM
&emsp;&emsp;传统的LSTM模型并没有考虑被评估的目标词和上下文的相互关系，为了引入这一部分的信息， TD-LSTM 应运而生。其基本思路是根据 target words 之前和之后的上下文分别构建两个 LSTM。本质上是模拟人类在 TDSA 任务中在目标词上下文寻找关键词。

<center><img src="C:\Users\HJHGJGHHG\Desktop\论文笔记与复现\Effective LSTMs for Target-Dependent Sentiment Classification/TD-LSTM.png"  style="zoom:30%;" width="100%"/></center>

&emsp;&emsp;$LSTM_L$ 的输入为**目标词之前的上下文加上目标词**，即从句子的第一个单词，到最后一个 target words  $W_{r-1}$ 依次输入；
&emsp;&emsp;$LSTM_R$ 的输入为**目标词之后的上下文加上目标词**，即从句子的最后一个单词  ，到第一个 target words  $W_{L+1}$依次输入。然后拼接两个输出，用 softmax 进行分类，损失函数为交叉熵。

## 三、TC-LSTM
&emsp;&emsp;为了加强目标词与上下文间的联系，先将 target words 字向量（当时应该是词向量）取平均得到 $V_{target}$，作为 target words 的代替。而后将 $V_{target}$ 与原来的词向量拼接。相较于 TD-LSTM，TC-LSTM 整合了target words与context words的相互关联信息。模型同样用 softmax 函数作为最后一层的激活函数来实现分类，用交叉熵作为损失函数来计算损失。
<center><img src="C:\Users\HJHGJGHHG\Desktop\论文笔记与复现\Effective LSTMs for Target-Dependent Sentiment Classification/TC-LSTM.png"  style="zoom:30%;" width="100%"/></center>

## 四、代码复现
### 1.数据集介绍
&emsp;&emsp;Twitter，具体说明见 data/readme.txt

### 2.结果：
