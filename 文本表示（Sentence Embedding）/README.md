# Sentence Embedding
&emsp;&emsp;**句向量**是**文本表示**中一个重要领域（私认为是文本表示的最终目的，词向量很多时候最后还是要生成句向量），文本表示广泛应用于召回、聚类、分类、匹配等等很多领域。评价句向量多用句间关系任务完成，如句间相似度、句间语义关系推理、问答对等等。  
&emsp;&emsp;我们将句向量模型分为 Representation-based（双塔式） 与 Interaction-based（交互式）两类。双塔式模型即用一个编码器分别给两个文本编码出句向量，然后把两个向量融合过一个浅层的分类器；交互式就是把两个文本一起输入进编码器，在编码的过程中让它们相互交换信息，再得到最终结果。如下图：
<center><img src="1.webp"  style="zoom:100%;" width="110%"/></center>

## Benchmark  
#### 1. Semantic Textual Similarity
* STS-B (SemEval 2017 Task 1: Semantic Textual Similarity - Multilingual and Cross-lingual Focused Evaluation)  [[home page]](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)  
* STS-12 (Semeval-2012 task 6: A pilot on semantic textual similarity)  [[home page]](https://aclanthology.org/S12-1051/)
* STS-13 (SEM 2013 shared task: Semantic Textual Similarity)  [[home page]](https://aclanthology.org/S13-1004/)
* STS-14 (SemEval-2014 task 10: Multilingual semantic textual similarity)  [[home page]](https://aclanthology.org/S14-2010/)
* STS-15 (SemEval-2015 task 2: Semantic textual similarity, English, Spanish and pilot on interpretability)  [[home page]](https://aclanthology.org/S15-2045/)
* STS-16 (SemEval-2016 task 1: Semantic textual similarity, monolingual and cross-lingual evaluation)  [[home page]](https://aclanthology.org/S16-1081/)
* SICK-R (SemEval-2014 task 1: valuation of compositional distributional semantic models on full sentences through semantic relatedness and textual entailment)  [[home page]](https://marcobaroni.org/composes/sick.html)
* MRPC (Microsoft Research Paraphrase Corpus)  [[home page]](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
