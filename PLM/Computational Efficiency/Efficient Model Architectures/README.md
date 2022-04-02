# Efficient Model Architectures
### [Notes](https://github.com/HJHGJGHHG/NLPPapers/blob/main/PLM/Computational%20Efficiency/Efficient%20Model%20Architectures/Notes.md)
## linearization (softmax)
#### 0. Overview
* (Survey)  **Efficient Transformers: A Survey**.  *Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler*  (preprint)  [[pdf]](https://arxiv.org/pdf/2009.06732.pdf)
* (Benchmark)  **Long Range Arena: A Benchmark for Efficient Transformers**.  *Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler*  [[pdf]](https://arxiv.org/pdf/2011.04006.pdf)
#### 1. Kernel method & Softmax Approximation
* **Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel**.  *Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, Ruslan Salakhutdinov*  (EMNLP 2019)  [[pdf]](https://arxiv.org/pdf/1908.11775v4.pdf)
* **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**.  *Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret*.  (ICML 2020)  [[pdf]](https://arxiv.org/pdf/2006.16236.pdf)  -***Linear Transformer***
* **Finetuning Pretrained Transformers into RNNs**.  *Jungo Kasai, Hao Peng, Yizhe Zhang, Dani Yogatama, Gabriel Ilharco, Nikolaos Pappas, Yi Mao, Weizhu Chen, Noah A. Smith*.  (EMNLP 2021)  [[EMNLP pdf]](https://aclanthology.org/2021.emnlp-main.830.pdf)
* **Rethinking Attention with Performers**.  *Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller*.  (ICLR 2021)  [[ICLR pdf]](https://openreview.net/pdf?id=Ua6zuk0WRH)  -***Performer***
* **Random Feature Attention**.  *Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A. Smith, Lingpeng Kong*.  (ICLR 2021)  [[pdf]](https://arxiv.org/pdf/2103.02143.pdf)  -***RFA***
* **cosFormer: Rethinking Softmax in Attention**.  *Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, Yiran Zhong*.  (ICLR 2022)  [[pdf]](https://arxiv.org/abs/2202.08791)  -***cosFormer***
#### 2. Low-Rank
* **Synthesizer: Rethinking Self-Attention in Transformer Models**.  *Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, Che Zheng.*  (ICML 2021)  [[pdf]](https://arxiv.org/pdf/2005.00743.pdf)  -***Synthesizer***
* **Linformer: Self-Attention with Linear Complexity**.  *Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma*.  [[ICLR pdf]](https://openreview.net/pdf?id=Bl8CQrx2Up4)  -***Linformer***

## Sparse Attention (attention matrix)
* **Generating Long Sequences with Sparse Transformers**.  *Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever*.  [[pdf]](https://paperswithcode.com/paper/190410509)  -***Sparse Transformer***
* **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**.  *Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov*  (ACL 2019)  [[ACL pdf]](https://aclanthology.org/P19-1285.pdf)  -***Transformer-XL***
* **Reformer: The Efficient Transformer**.  *Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya*.  (ICLR 2020)  [[ICLR pdf]](https://openreview.net/pdf?id=rkgNKkHtvB)  -***Reformer***
* **Longformer: The Long-Document Transformer**.  *Iz Beltagy, Matthew E. Peters, Arman Cohan*.  [[pdf]](https://arxiv.org/pdf/2004.05150v2.pdf)  -***Longformer***
* **Big Bird: Transformers for Longer Sequences**.  *Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed*.  (NeurIPS 2020)  [[NeurIPS pdf]](https://proceedings.neurips.cc//paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)  -***BigBird***

## Conditional computation
* **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**.  *Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean*.  [[pdf]](https://arxiv.org/pdf/1701.06538v1.pdf)  -***MoE***