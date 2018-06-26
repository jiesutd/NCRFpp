## Hyperparamter tuning

1. If you use large batch, you'd better set `avg_batch_loss` to `True`
2. You can get 2~5 points' improvement if you use 300-d word embedding instead of 50-d word embedding
3. If you want to write a script to tune hyperparameters, you can use the `main_parse.py` to set hyperparameters in command line arguements
4. `lr` needs to be carefully tuned for different structures:
    * If you run LSTM-LSTM-CRF on CONLL-2003 dataset, a good `lr` is 0.015
    * If you run LSTM-CNN-CRF on CONLL-2003 dataset,, a good `lr` is 0.005