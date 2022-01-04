# BERT & RoBERTa
&emsp;&emsp;使用经典 BERT 与 RoBERTa 架构做相似度任务。实验结果：

| 模型与策略 | STS-B| MRPC  |
|  :--:  |  :--:   |  :--:   |
|  BERT-Base  CLS  | 88.1 |     |
|  RoBERTa-Base  CLS  | 88.1 |     |
|  BERT-Base  last hidden + mean  | 88.1 |     |
|  RoBERTa-Base  last hidden + mean  | 88.1 |     |

&emsp;&emsp;注：STS-B 指标为 Spearman correlation，MRPC指标为 F1。BERT 模型均为 uncased。
&emsp;&emsp;策略：
* CLS：最后一层 CLS 的 Embedding 作为句向量；
* last hidden + mean： 最后一层序列的输出(各词向量)求平均