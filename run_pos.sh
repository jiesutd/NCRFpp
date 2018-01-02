python main.py --status train \
		--wordemb glove \
		--train ../data/wsj_pos/train.pos \
		--dev ../data/wsj_pos/dev.pos \
		--test ../data/wsj_pos/test.pos \
		--savemodel ../data/wsj_pos/saved_model \
		--seg False \


# python train_batch.py --status decode \
# 		--raw data/$1/dev.bmes \
# 		--savedset data/$1/saved_model.lstmcrf.dset \
# 		--loadmodel data/$1/saved_model.lstmcrf.13.model \
# 		--output data/$1/raw.out \
