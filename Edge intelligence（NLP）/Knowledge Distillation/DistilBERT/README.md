# DistilBERT：预训练中融入蒸馏思想
&emsp;&emsp;BERT 本身有较多的transformer层，这也是 BERT 比较强的核心原因之一，有了上面 DistilLSTM 的经验，我们不禁要开始想一个问题，BERT 里面的Transformer 是不是也可能没有被充分学习呢？在更多层训练完的 transformer（原版BERT）的支持下少几层的 BERT 是否仍然保持比较好的效果呢？本文给出了肯定的答案。  
&emsp;&emsp;模型本身没什么改动：  
* 在12层 Transformer-encoder 的基础上每2层中去掉一层，最终将12层减少到了6层。
* 去掉了token type embedding 和 pooler。
* 利用 teacher model 的参数来初始化 student。

&emsp;&emsp;Loss 由三部分组成：  
* $L_{ce}$：与 Hinton 的论文一样，是两个模型广义 softmax 输出的交叉熵；
* $L_{mlm}$：常规 MLM 训练损失；
* $L_{cos}$：两模型隐层间损失。（？？？说的不明不白的，文章中只有一句话："...add a cosine embedding loss ($L_{cos}$) which will tend to align the directions of the student and teacher hidden states vectors. "。所以到底是 last hidden state 还是中间层的 state？）

&emsp;&emsp;预训练采用的是 RoBERTa 的方法：大 Batchsize，动态 Mask，去掉 NSP 