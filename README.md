NCRF++: An Open-source Neural Sequence Labeling Toolkit
======
Sequence labeling models are quite popular in many NLP tasks, such as Named Entity Recognition (NER), part-of-speech (POS) tagging and word segmentation. State-of-the-art sequence labeling models mostly utilize the CRF structure with input word features. LSTM (or bidirectional LSTM) is a popular deep learning based feature extractor in sequence labeling task. And CNN can also be used due to faster computation. Besides, features within word are also useful to represent word, which can be captured by character LSTM or character CNN structure or human-defined neural features.

NCRF++ is a PyTorch based framework with flexiable choices of input features and output structures. The design of neural sequence labeling models with NCRF++ is fully configurable through a configuration file, which does not require any code work. NCRF++ is a neural version of [CRF++](http://taku910.github.io/crfpp/), which is a famous statistical CRF framework. The detailed experiment report using NCRF++ has been accepted at COLING 2018.

NCRF++ supports diffent structure combinations of on three levels: character sequence representation, word sequence representation and inference layer.

* Character sequence representation: character LSTM, character GRU, character CNN and handcrafted word features.
* Word sequence representation: word LSTM, word GRU, word CNN.
* Inference layer: Softmax, CRF.


Welcome to star this repository!

Requirement:
======
	Python: 2.7   
	PyTorch: 0.3 (currently not support 0.4, will update soon)


Advantages
========
* 1.Fully configurable: all the neural model structures can be setted with a configuration file.
* 2.State-of-the-art system performance: models build on NCRF++ can give comparable or better results compared with state-of-the-art models.
* 3.Flexible with features: user can define their own features and pretrained feature embeddings.
* 4.Fast running speed: NCRF++ utilizes fully batched operations, making the system efficient with the help of GPU (>1000sent/s for training and >2000sents/s for decoding).
* 5.N best output: NCRF++ support `nbest` decoding (with their probabilities).


1.Usage
=========
NCRF++ supports designing the neural network structure through a configuration file. The program can run in two status; ***training*** and ***decoding***. (sample configuration and data have been included in this repository)  

In ***training*** status:
`python main.py --config demo.train.config`

In ***decoding*** status:
`python main.py --config demo.decode.config`

The configuration file controls the network structure, I/O, training setting and hyperparameters. 

***Detail configurations and explanations are listed [here](readme/Configuration.md).***

NCRF++ is designed in three layers (shown below): character sequence layer; word sequence layer and inference layer. By using the configuration file, most of the state-of-the-art models can be easily replicated ***without coding***. On the other hand, users can extend each layer by designing their own modules (for example, they may want to design their own neural structures other than CNN/LSTM/GRU). Our layer-wised design makes the module extension convenient, the instruction of module extension can be found [here](readme/Extension.md).

![alt text](readme/architecture.png "Layer-size design")

2.Performance
=========
Results on CONLL 2003 English NER task are better or comparable with SOTA results with the same structures. 

CharLSTM+WordLSTM+CRF: 91.20 vs 90.94 of [Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf);

CharCNN+WordLSTM+CRF:  91.35 vs 91.21 of [Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf).   

In default, `LSTM` is bidirectional LSTM.    

|ID| Model | Nochar | CharLSTM |CharCNN   
|---|--------- | --- | --- | ------    
|1| WordLSTM | 88.57 | 90.84 | 90.73  
|2| WordLSTM+CRF | 89.45 | **91.20** | **91.35** 
|3| WordCNN |  88.56| 90.46 | 90.30  
|4| WordCNN+CRF |  88.90 | 90.70 | 90.43  

We have compared twelve neural sequence labeling models (`{charLSTM, charCNN, None} x {wordLSTM, wordCNN} x {softmax, CRF}`) on three benchmarks (POS, Chunking, NER) under statistical experiments, detail results and comparisons can be found in our COLING 2018 paper (coming soon).
 

3.External feature defining
=========
NCRF++ has integrated several SOTA neural characrter sequence feature extractors: CNN ([Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf)), LSTM ([Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf)) and GRU ([Yang .etc, ICLR17](https://arxiv.org/pdf/1703.06345.pdf)). In addition, handcrafted features have been proven important in sequence labeling tasks. NCRF++ allows users designing their own features such as Capitalization, POS tag or any other features (grey circles in above figure). Users can configure the self-defined features through configuration file (feature embedding size, pretrained feature embeddings .etc). The sample input data format is given at [train.cappos.bmes](sample_data/train.cappos.bmes), which includes two human-defined features `[POS]` and `[Cap]`. (`[POS]` and `[Cap]` are two examples, you can give your feature any name you want, just follow the format `[xx]` and configure the feature with the same name in configuration file.)
User can configure each feature in configuration file by using 

```Python
feature=[POS] emb_size=20 emb_dir=%your_pretrained_POS_embedding
feature=[Cap] emb_size=20 emb_dir=%your_pretrained_Cap_embedding
```

Feature without pretrained embedding will be randomly initialized.


4.Speed
=========
NCRF++ is implemented using fully batched calculation, making it quite effcient on both model training and decoding. With the help of GPU (Nvidia GTX 1080) and large batch size, LSTMCRF model built with NCRF++ can reach 1000 sents/s and 2000sents/s on training and decoding status, respectively.

![alt text](readme/speed.png "System speed on NER data")


5.N best decoding performance:
=========
Traditional CRF structure decodes only one label sequence with largest probabolities (i.e. 1-best output). While NCRF++ can give a large choice, it can decode `n` label sequences with the top `n` probabilities (i.e. n-best output). The nbest decodeing has been supported by several popular **statistical** CRF framework. However to the best of our knowledge, NCRF++ is the only and the first toolkit which support nbest decoding in **neural** CRF models. 

In our implementation, when the nbest=10, CharCNN+WordLSTM+CRF model built in NCRF++ can give 97.47% oracle F1-value (F1 = 91.35% when nbest=1) on CoNLL 2003 NER task.

![alt text](readme/nbest.png  "N best decoding oracle result")


Cite: 
========
If you use experiments results of NCRF++, please cite our COLING paper:

    @article{yang2018design,  
     title={Design Challenges and Misconceptions in Neural Sequence Labeling},  
     author={Jie Yang, Shuailong Liang and Yue Zhang},  
     booktitle={Proceedings of the 27th International Conference on Computational Linguistics (COLING)},
     year={2018}  
    }


Updating...
====
* 2018-Mar-30, NCRF++ v0.1, initial version
* 2018-Jan-06, add result comparison.
* 2018-Jan-02, support character feature selection. 
* 2017-Dec-06, init version

