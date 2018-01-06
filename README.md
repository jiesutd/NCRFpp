Neural Sequence labeling model implemented using PyTorch
======
Requirement:
======
	Python: 2.7   
	PyTorch: 0.3

Functions
========
* State-of-the-art sequence labeling model with the combination of different input features(word embedding, char LSTM, char CNN ) and output structures (CRF, softmax). i.e. (BILSTM-CRF, BiLSTM) x (character LSTM, character CNN, None).
* Batch training/decoding supported, with fast running speed. Most model cost less than 1 minitus for each epoch, finish the whole training process with in 2 hours (with the help of GPU).
* Character feature can be selected within LSTM, CNN or None (through `data.char_features="CNN"/"LSTM"` or `data.HP_use_char = False` in file `main.py`).
* Output structure can be choosed in CRF or softmax (by using `from model.bilstmcrf import BiLSTM_CRF as SeqModel` or `from model.bilstm import BiLSTM as SeqModel` in file `main.py`).
* Bidirectional LSTM-CRF with character features give the best result on Named Entity Recognition (NER) task. 

Performance
=========
Results on CONLL 2003 English NER task are better or comparable with SOTA results with same structures.    
In default, `LSTM` means bidirectional lstm structure.    

|ID| Model | Dev | Test |Note   
|---|--------- | --- | --- | ------    
|1| LSTM | 93.12 | 88.74 |   
|2| CharLSTM+LSTM | 94.31 | 90.52 |   
|3| CharCNN+LSTM |  94.41| 90.37 |   
|4| LSTMCRF |  93.34 | 89.48 |   
|5| CharLSTM+LSTMCRF | 94.77 | **91.33** |    
|6| CharCNN+LSTMCRF | 94.83 | **91.22** |    
|7| Lample .etc, NAACL16 | 	| 90.94 | same structure with 5   
|8| Xuezhe Ma .etc, ACL16 | 	| 91.21 | same structure with 6   



Updating...
====
* 2018-Jan-06, add result comparison.
* 2018-Jan-02, support character feature selection. 
* 2017-Dec-06, init version

