python main.py --status train \
		--wordemb richchar \
		--train ../data/onto4ner.cn/train.char.bmes \
		--dev ../data/onto4ner.cn/dev.char.bmes \
		--test ../data/onto4ner.cn/test.char.bmes \
		--savemodel ../data/onto4ner.cn/saved_model \


# python train_batch.py --status decode \
# 		--raw data/$1/dev.bmes \
# 		--savedset data/$1/saved_model.lstmcrf.dset \
# 		--loadmodel data/$1/saved_model.lstmcrf.13.model \
# 		--output data/$1/raw.out \
