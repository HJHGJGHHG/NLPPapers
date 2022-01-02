# Memory Network
&emsp;&emsp;基于论文：
* 《MEMORY NETWORKS》https://arxiv.org/abs/1410.3916v11
## 一、开篇之作
&emsp;&emsp;Memory Network是深度学习的一个小分支，从2014年被提出到现在也逐渐发展出了几个成熟的模型，我们先从开篇之作 FaceBook 的《MEMORY NETWORKS》说起。
&emsp;&emsp;传统的深度学习模型（RNN、LSTM、GRU等）使用 hidden states 或者 Attention 机制作为他们的记忆功能，但是这种方法产生的记忆太小了，无法精确记录一段话中所表达的全部内容，在通过全连接层时损失了很多信息。鉴于此该文章提出了一种可读写的外部记忆模块，并将其和 inference 组件联合训练，最终得到一个可以被灵活操作的记忆模块。
