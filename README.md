NCRF++: An Open-source Neural Sequence Labeling Toolkit
======
State-of-the-art sequence labeling models mostly utilize the CRF structure with input word features. LSTM (or bidirectional LSTM) is a popular deep learning based feature extractor in sequence labeling task. And CNN can also be used due to faster computation. Besides, features within word are also useful to represent word, which can be captured by character LSTM or character CNN structure or human-defined neural features.

NCRF++ is a PyTorch based framework with flexiable choices of input features and output structures. Designing neural sequence labeling models in NCRF++ is fully configurable through a configuration file, which does not require any code work. NCRF++ is a neural version of [CRF++](http://taku910.github.io/crfpp/), which is a famous statistical CRF framework.

NCRF++ supports diffent structure combinations of on three levels: character sequence representation, word sequence representation and inference layer.

* Character sequence representation: character LSTM, character GRU, character CNN and handcrafted word features.
* Word sequence representation: word LSTM, word GRU, word CNN.
* Inference layer: Softmax, CRF.


Welcome to star this repository!

Requirement:
======
	Python: 2.7   
	PyTorch: 0.3.0


Advantages
========
* 1.Fully configurable: all the neural model structure can be setted using a configuration file.
* 2.State-of-the-art system performance: models build on NCRF++ can give comparable or better results compared with SOTA models.
* 3.Flexible with features: user can define their own features and pretrained feature embeddings.
* 4.Fast running speed: NCRF++ utilizes fully batched operations, making the system efficient with the help of GPU (>1000sent/s for training and >2000sents/s for decoding).
* 5.N best output: NCRF++ support nbest decoding. 


1.Usage
=========
NCRF++ supports designing the neural network structure through configuration file. The program can run in two status; training and decoding. (sample configuration and data have been included in this repository)  

In training status:
`python main.py --config demo.train.config`

In decoding status:
`python main.py --config demo.decode.config`

The configuration file controls the network structure, I/O, training setting and hyperparameters. Details configurations are list [here](readme/Configuration.md). 

NCRF++ is designed in three layers: character sequence layer; word sequence layer and inference layer. By using the configuration file, most of the state-of-the-art models can be easily replicated without coding. On the other hand, users can extend each layer by designing their own modules (for example, they may want to design their own neural structures other than CNN/LSTM/GRU). Our layer-wised design makes the module extension convenient, the instruction of module extension can be found [here](readme/Extension.md).

2.Performance
=========
Results on CONLL 2003 English NER task are better or comparable with SOTA results with same structures. 

CharLSTM+WordLSTM+CRF: 91.20 vs 90.94 of [Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf);

CharCNN+WordLSTM+CRF:  91.26 vs 91.21 of [Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf).   

In default, `LSTM` means bidirectional lstm structure.    

|ID| Model | Nochar | CharLSTM |CharCNN   
|---|--------- | --- | --- | ------    
|1| WordLSTM | 88.57 | 90.84 | 90.73  
|2| WordLSTM+CRF | 89.45 | **91.20** | **91.26** 
|3| WordCNN |  88.56| 90.46 | 90.30  
|4| WordCNN+CRF |  88.90 | 90.70 | 90.43  


3.External feature defining
=========
NCRF++ has integrated several SOTA neural characrter sequence feature extractors: CNN ([Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf)), LSTM ([Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf)) and GRU ([Yang .etc, ICLR17](https://arxiv.org/pdf/1703.06345.pdf)). In addition, handcrafted features have been proven important in sequence labeling tasks. NCRF++ allows users designing their own features such as Capitalization, POS tag or any other features. Users can configure the self-defined features through configuration file (feature embedding size, pretrained feature embeddings .etc). The sample input data format is given at [train.cappos.bmes](sample_data/train.cappos.bmes), which includes two human-defined features `[POS]` and `[Cap]`.
User can configure each feature in configuration file by using 

```Python
feature=[POS] emb_size=20 emb_dir=%your_pretrained_POS_embedding
feature=[Cap] emb_size=20 emb_dir=%your_pretrained_Cap_embedding
```

Feature without pretrained embedding will be randomly initialized.


4.Speed
=========
NCRF++ is implemented using fully batched calculation, making it quite effcient on both model training and decoding. With the help of GPU (Nvidia GTX 1080) and large batch size, NCRF++ can reach 1000 sents/s and 2000sents/s on training and decoding status, respectively.

![alt text](readme/speed.png "System speed on NER data")


5.N best decoding performance:
=========
Traditional CRF structure decodes only one label sequence with largest probabolities (i.e. 1-best output). While NCRF++ can give a large choice, it can decode `n` label sequences with the top `n` probabilities (i.e. n-best output). The nbest decodeing has been supported by several popular **statistical** CRF framework. However to the best of our knowledge, NCRF++ is the only and the first toolkit which support nbest decoding in **neural** CRF models. 

In our implementation, when the nbest=10, CharCNN+WordLSTM+CRF model built in NCRF++ can give 97.47% oracle F1-value on CoNLL 2003 NER task.

![alt text](readme/nbest.png  "N best decoding oracle result")


Updating...
====
* 2018-Mar-30, NCRF++ v0.1, initial version
* 2018-Jan-06, add result comparison.
* 2018-Jan-02, support character feature selection. 
* 2017-Dec-06, init version

