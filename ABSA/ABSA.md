# ABSAPapers
&emsp;&emsp;Papers and related resources on aspect-based sentiment analysis (ABSA). This repository mainly focused on aspect-term sentiment classification (ATSC). ABSA task contains five fine-grained subtasks:
- Aspect Term Sentiment Classification (ATSC)
- Aspect Term Extraction (ATE)
- Aspect Category Sentiment Classification (ACSC)
- Aspect Category Detection (ACD)
- Opinion Term Extraction (OTE)

&emsp;&emsp;Notes and annotations are mainly based on Chinese.

---

&emsp;&emsp;读过的方面级情感分析论文与相关资源。这里主要关注方面词（aspect-term）的情感分类。具体来说，方面级情感分析包括方面词情感分类、方面词抽取、方面类目情感分类、方面类目抽取、观点词抽取五个子任务。
&emsp;&emsp;主要包含笔记版pdf原文，个人思考。部分文章有复现代码，主要参考 https://github.com/songyouwei/ABSA-PyTorch ，修补了一些错误并加上了注释。为了精简考虑，没有将所有模型放在一个项目中，尽量保证一个模型一个项目，增加可读性。

## 一、Before BERT: Embedding Method
- **Effective LSTMs for Target-Dependent Sentiment Classification**. *Duyu Tang, Bing Qin, Xiaocheng Feng, Ting Liu*. (COLING 2016) [[paper]](https://www.aclweb.org/anthology/C16-1311)[[code]](https://drive.google.com/drive/folders/17RF8MZs456ov9MDiUYZp0SCGL6LvBQl6) - ***TD-LSTM TC-LSTM*** [[my code]](https://github.com/HJHGJGHHG/NLPPapers/tree/main/ABSA/Effective%20LSTMs%20for%20Target-Dependent%20Sentiment%20Classification/code)