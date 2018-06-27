## Hyperparamter tuning

1. If you use large batch (e.g. batch_size > 100), you'd better set `avg_batch_loss=True`.  to get a stable training process. For small batch size, `avg_batch_loss=True` will converge faster and sometimes gives better performance (e.g. CoNLL 2003 NER).
2. You can get 2~5 points' improvement on the CONLL2012 dataset if you use 300-d pretrained word vectors [here](https://nlp.stanford.edu/projects/glove/) instead of 50-d pretrained word vectors.
3. If you want to write a script to tune hyperparameters, you can use the `main_parse.py` to set hyperparameters in command line arguements.
4. `lr` needs to be carefully tuned for different structures:
    * If you run LSTM-LSTM-CRF on CONLL-2003 dataset, a reasonable `lr` is 0.015.
    * If you run LSTM-CNN-CRF on CONLL-2003 dataset,, a reasonable `lr` is 0.005.
    * If you run CNN-CNN-CRF on CONLL-2003 dataset, a reasonable `lr` is 0.0015.
    * Note: `lr` above does not guarantee the best performance but only a reasonable performance (~90% for F1-score). You still need to tune the hyperparameters if you want to reach the best performance.