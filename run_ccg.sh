python main.py --status train \
		--wordemb glove \
		--train ../data/ccg/train.ccg \
		--dev ../data/ccg/dev.ccg \
		--test ../data/ccg/test.ccg \
		--savemodel ../data/ccg/saved_model \
		--seg False \


# python train_batch.py --status decode \
# 		--raw data/$1/dev.bmes \
# 		--savedset data/$1/saved_model.lstmcrf.dset \
# 		--loadmodel data/$1/saved_model.lstmcrf.13.model \
# 		--output data/$1/raw.out \
