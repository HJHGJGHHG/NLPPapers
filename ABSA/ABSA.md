# ABSAPapers
&emsp;&emsp;Papers and related resources on aspect-based sentiment analysis (ABSA). This repository mainly focused on aspect-term sentiment classification (ATSC). ABSA task contains five fine-grained subtasks:
- Aspect Term Sentiment Classification (ATSC)
- Aspect Term Extraction (ATE)
- Aspect Category Sentiment Classification (ACSC)
- Aspect Category Detection (ACD)
- Opinion Term Extraction (OTE)

&emsp;&emsp;Notes and annotations are mainly based on Chinese.

---

&emsp;&emsp;读过的基于方面的情感分析论文与相关资源。这里主要关注方面词（aspect-term）的情感分类。具体来说，方面级情感分析包括方面词情感分类、方面词抽取、方面类目情感分类、方面类目抽取、观点词抽取五个子任务。
&emsp;&emsp;主要包含笔记版pdf原文，个人思考。部分文章有复现代码，主要参考 https://github.com/songyouwei/ABSA-PyTorch ，修补了一些错误并加上了注释。为了精简考虑，没有将所有模型放在一个项目中，尽量保证一个模型一个项目，增加可读性。
## 一、几个概念的定义
&emsp;&emsp;ABSA 中的几个概念还是有点模糊的，容易混淆。以下是**个人见解**：
&emsp;&emsp;例句：“这家饭店菜不错，但服务太差。”

* 实体(Entity)：抽象或客观存在的概念体。（饭店）
* 方面(Aspect)：刻画实体不同方面的词。(饭菜口味、服务、大小、位置 ……)
* 目标(Target)：一般情况下指实体。
&emsp;&emsp;我们现在可以看出 aspect 和 target 的区别了。**target** 一般指文中的描述实体，是**粗粒度任务**，而且**基本上会直接出现在文本中**；而 **aspect** 则为**细粒度任务**，值同一 target 的不同方面，而且**不一定会直接出现在文本中**。我们所说的 ABSA 通常指 aspect-level sentiment analysis。target 仅在部分任务中提到，如存在多个 target 的文本。
&emsp;&emsp;对于 aspect，根据具体的任务又分为 aspect term 和 aspect category。
* aspect term：**文本中刻画实体的属性词**。如上文中的句子，aspect term 就只有口味与服务。
* aspect category：**预先给定的属性词的集合**。如任务预先给定的 aspect category 为：{口味、服务、大小、位置}，则文本中只有对前两者的情感极性，后两者则不涉及。

&emsp;&emsp;可以结合 SemEval 2014 Task 4 的数据集理解：
1. SB1: **Aspect term extraction**(ATE): 这个任务是识别句子中的aspect term。
2. SB2: **Aspect term polarity**(ATSC)：这个任务是在给定aspect term的情况下判断aspect term情感极性(positive/negative/conflict/neutral) 当句子对aspect term表达的情感既有积极又有消极的时候该aspect term的情感极性为conflict。
3. SB3: **Aspect category detection**(ACD)：这个任务是首先预定义一个aspect categories的集合比如“price,food”等，之后判断哪一个aspect出现在了句子中。例如，句子“Delicious but expensive”中 food 和 price 并没有显示出现，但是可以通过 delicious 和 expensive 来推断出来。
4. SB4: **Aspect category polarity**(ACSC)：这个任务是对于一个句子给定aspect categories，之后判断该aspect category的情感极性。

---

## 二、Before BERT: Embedding Method
- **Effective LSTMs for Target-Dependent Sentiment Classification**. *Duyu Tang, Bing Qin, Xiaocheng Feng, Ting Liu*. (COLING 2016) [[paper]](https://www.aclweb.org/anthology/C16-1311)[[code]](https://drive.google.com/drive/folders/17RF8MZs456ov9MDiUYZp0SCGL6LvBQl6) - ***TD-LSTM TC-LSTM***  [[my code]](https://github.com/HJHGJGHHG/NLPPapers/tree/main/ABSA/ABSA)
- **Attention-based LSTM for Aspect-level Sentiment Classification**. *Yequan Wang, Minlie Huang, Xiaoyan Zhu, Li Zhao*. (EMNLP 2016) [[paper]](https://aclweb.org/anthology/D16-1058) - ***ATAE-LSTM***  [[my code]](https://github.com/HJHGJGHHG/NLPPapers/tree/main/ABSA/ABSA)
- **Aspect Level Sentiment Classification with Deep Memory Network**. *Duyu Tang, Bing Qin, Ting Liu*. (EMNLP 2016) [[paper]](https://www.aclweb.org/anthology/D16-1021)[[code]](https://drive.google.com/drive/folders/1Hc886aivHmIzwlawapzbpRdTfPoTyi1U) - ***MemNet***
