## Hyperparamter tuning on CoNLL 2003 English NER task

1. If you use large batch (e.g. batch_size > 100), you'd better set `avg_batch_loss=True` to get a stable training process. For small batch size, `avg_batch_loss=True` will converge faster and sometimes gives better performance (e.g. CoNLL 2003 NER).
2. You can get better performance on the CoNLL 2003 English dataset if you use 100-d pretrained word vectors [here](https://nlp.stanford.edu/projects/glove/) instead of 50-d pretrained word vectors.
3. If you want to write a script to tune hyperparameters, you can use the `main_parse.py` to set hyperparameters in command line arguements.
4. Model performance is sensitive with `lr` which needs to be carefully tuned under different structures:
    * Word level LSTM models (e.g. char LSTM + word LSTM + CRF) would prefer a `lr` around 0.015.
    * Word level CNN models (e.g. char LSTM + word CNN + CRF) would prefer a `lr` around 0.005 and with more iterations.
    * You can refer the COLING paper "[Design Challenges and Misconceptions in Neural Sequence Labeling](https://arxiv.org/pdf/1806.04470.pdf)" for more hyperparameter settings.
