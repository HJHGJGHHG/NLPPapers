# BERT 词向量的各向异性
&emsp;&emsp;本文聊聊使用 BERT 抽取的词向量（无监督）有哪些问题。（其实看到这个问题有点迷糊，BERT 用于匹配不是交互式且端到端的么...应该将两个句子拼接喂给模型直接得到相似度啊...讨论无监督抽取词向量是不是不太 fair...anyway 既然大佬都说能这么做了，那就能这么做吧...）  
&emsp;&emsp;承载三篇论文：
* [Representation Degeneration Problem in Training Natural Language Generation Models](https://github.com/HJHGJGHHG/NLPPapers/blob/main/%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%EF%BC%88Sentence%20Embedding%EF%BC%89/Analysis/BERT%E8%AF%8D%E5%90%91%E9%87%8F%E5%90%84%E5%90%91%E5%BC%82%E6%80%A7/Representation%20Degeneration%20Problem%20in%20Training%20Natural%20Language%20Generation%20Models.pdf)
* [Improving Neural Language Generation with Spectrum Control](https://github.com/HJHGJGHHG/NLPPapers/blob/main/%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%EF%BC%88Sentence%20Embedding%EF%BC%89/Analysis/BERT%E8%AF%8D%E5%90%91%E9%87%8F%E5%90%84%E5%90%91%E5%BC%82%E6%80%A7/Improving%20Neural%20Language%20Generation%20with%20Spectrum%20Control.pdf)
* [Isotropy in the Contextual Embedding Space: Clusters and Manifolds]()
## 一、Representation Degeneration Problem in Training Natural Language Generation Models