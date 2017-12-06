python main.py --status train \
		--embedding glove \
		--train data/$1/train.bmes \
		--dev data/$1/dev.bmes \
		--test data/$1/test.bmes \
		--savemodel data/$1/saved_model.lstmcrf \


# python train_batch.py --status decode \
# 		--raw data/$1/dev.bmes \
# 		--savedset data/$1/saved_model.lstmcrf.dset \
# 		--loadmodel data/$1/saved_model.lstmcrf.13.model \
# 		--output data/$1/raw.out \
