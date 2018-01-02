Neural CRF model implemented using PyTorch
======
requirement:
	Python: 2.7 
	Pytorch: 0.3

Performance
=========
BILSTM-CRF with character lstm features

Batch training/decoding supported.

Character feature can be select using LSTM/CNN or None, through data.char_features="CNN"/"LSTM" or data.HP_use_char = False .

On CONLL 2003 English data:
F1 dev: 94.7%
F1 test: 90.9% +- 0.2%


Updating...
====
* 2018-Jan-02, support character feature selection. 
* 2017-Dec-06, init version

