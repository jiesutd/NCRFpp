# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-29 14:21:39
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
from .ncrf_transformers import NCRFTransformers

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        if (not data.use_word_seq) and (data.high_level_transformer == None or data.high_level_transformer == "None"):
            print("ERROR: at least one of use_word and high_level_transformer should be valid")
            exit(0)
        self.gpu = data.HP_gpu
        self.device = data.device
        self.words2sent = data.words2sent_representation.upper()
        self.use_char = data.use_char
        self.use_word_seq = data.use_word_seq
        self.use_word_emb = data.use_word_emb
        self.dropout = nn.Dropout(data.HP_dropout).to(self.device)
        self.wordrep = WordRep(data)
        self.input_size = 0 #data.word_emb_dim  ## input size of upper layer
        self.feature_num = data.feature_num
        self.output_hidden_dim = data.HP_hidden_dim
        if self.use_word_seq:
            if not data.silence:
                print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
            if self.use_char:
                self.input_size += data.HP_char_hidden_dim
                if data.char_feature_extractor == "ALL":
                    self.input_size += data.HP_char_hidden_dim
            if self.use_word_emb:
                self.input_size += data.word_emb_dim
            self.low_level_transformer = data.low_level_transformer
            if self.low_level_transformer != None and self.low_level_transformer.lower() != "none":
                self.input_size += 768 ## maybe changed based on the choice of BERT model
            
            if not data.sentence_classification:
                for idx in range(self.feature_num):
                    self.input_size += data.feature_emb_dims[idx]
            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.word_feature_extractor = data.word_feature_extractor
            if self.word_feature_extractor == "GRU" or self.word_feature_extractor == "LSTM":
                if data.HP_bilstm :
                    rnn_hidden = data.HP_hidden_dim // 2
                else:
                    rnn_hidden = data.HP_hidden_dim
                if self.word_feature_extractor == "LSTM":
                    self.lstm = nn.LSTM(self.input_size, rnn_hidden, num_layers=data.HP_lstm_layer, batch_first=True, bidirectional=data.HP_bilstm).to(self.device)
                elif self.word_feature_extractor == "GRU":
                    self.lstm = nn.GRU(self.input_size, rnn_hidden, num_layers=data.HP_lstm_layer, batch_first=True, bidirectional=sdata.HP_bilstm).to(self.device)
            elif self.word_feature_extractor == "FF":
                self.ff = nn.Linear(self.input_size, data.HP_hidden_dim).to(self.device)
            elif self.word_feature_extractor == "CNN":
                # cnn_hidden = data.HP_hidden_dim
                self.word2cnn = nn.Linear(self.input_size, data.HP_hidden_dim).to(self.device).to(self.device)
                self.cnn_layer = data.HP_cnn_layer
                print("CNN layer: ", self.cnn_layer)
                self.cnn_list = nn.ModuleList()
                self.cnn_drop_list = nn.ModuleList()
                self.cnn_batchnorm_list = nn.ModuleList()
                kernel = 3
                pad_size = int((kernel-1)/2)
                for idx in range(self.cnn_layer):
                    self.cnn_list.append(nn.Conv1d(data.HP_hidden_dim, data.HP_hidden_dim, kernel_size=kernel, padding=pad_size).to(self.device))
                    self.cnn_drop_list.append(nn.Dropout(data.HP_dropout).to(self.device))
                    self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.HP_hidden_dim).to(self.device))
        else:
            self.output_hidden_dim = 0

        ## set high level transformer
        self.high_level_transformer = data.high_level_transformer 
        if self.high_level_transformer != None and self.high_level_transformer != "None":
            self.high_level_transformer_finetune = data.high_level_transformer_finetune
            self.high_transformer = NCRFTransformers(self.high_level_transformer, self.device)
            self.output_hidden_dim += 768 ## Todo: dim may change with the choice of BERT models

        ## aggregate  word to sentence
        if self.words2sent == "ATTENTION" or self.words2sent == "ATT":
            self.word_weights = nn.Linear(self.output_hidden_dim, 1).to(self.device)
        # The linear layer that maps from hidden state space to tag space
        if not data.sentence_classification and data.use_crf:
            self.hidden2tag = nn.Linear(self.output_hidden_dim, data.label_alphabet_size+2).to(self.device)
        elif data.sentence_classification:
            ## add feature dim if classification
            self.feature_num = data.feature_num
            self.feature_embedding_dims = data.feature_emb_dims
            self.feature_embeddings = nn.ModuleList()
            feature_dim_size = 0 
            for idx in range(self.feature_num):
                self.feature_embeddings.append(nn.Embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx]).to(self.device))
                feature_dim_size += self.feature_embedding_dims[idx]
            for idx in range(self.feature_num):
                if data.pretrain_feature_embeddings[idx] is not None:
                    self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(data.pretrain_feature_embeddings[idx]))
                else:
                    self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(self.random_embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx])))
            self.hidden2tag = nn.Linear(self.output_hidden_dim+feature_dim_size, data.label_alphabet_size).to(self.device)
        else: 
            self.hidden2tag = nn.Linear(self.output_hidden_dim, data.label_alphabet_size).to(self.device)


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb



    def network_out_features(self, *input):
        word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_word_text = input[:7]
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        if self.use_word_seq:
            word_represent = self.wordrep(*input)
            ## word_embs (batch_size, seq_len, embed_size)
            if self.word_feature_extractor == "CNN":
                batch_size = word_inputs.size(0)
                word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
                for idx in range(self.cnn_layer):
                    if idx == 0:
                        cnn_feature = F.relu(self.cnn_list[idx](word_in))
                    else:
                        cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                    cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                    if batch_size > 1:
                        cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
                feature_out = cnn_feature.transpose(2,1).contiguous()
            elif self.word_feature_extractor == "FF":
                feature_out = torch.tanh(self.ff(word_represent)).transpose(2,1).contiguous()
            elif self.word_feature_extractor == None or self.word_feature_extractor.lower() == "none" :
                feature_out = word_represent
            else:
                packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
                hidden = None
                lstm_out, hidden = self.lstm(packed_words, hidden)
                lstm_out, _ = pad_packed_sequence(lstm_out)
                ## lstm_out (seq_len, seq_len, hidden_size)
                feature_out = lstm_out.transpose(1,0)
            ## feature_out (batch_size, seq_len, hidden_size)
            feature_out = feature_out.contiguous()
            ## merge bert features
            if self.high_level_transformer != None and self.high_level_transformer != "None":
                if self.training and self.high_level_transformer_finetune:
                    self.high_transformer.train()
                    transformer_output = self.high_transformer.extract_features(batch_word_text)
                else:
                    self.high_transformer.eval()
                    with torch.no_grad():
                        transformer_output = self.high_transformer.extract_features(batch_word_text)
                feature_out = torch.cat([feature_out, transformer_output], 2)
        else:
            if self.high_level_transformer != None and self.high_level_transformer.lower() != "none":
                feature_out = self.high_transformer.extract_features(batch_word_text)                
        return feature_out
        

    def forward(self, *input):
        feature_out = self.network_out_features(*input) 
        feature_out = self.dropout(feature_out)
        ## feature_out (batch_size, seq_len, hidden_size)
        outputs = self.hidden2tag(feature_out)
        return outputs


    def sentence_representation(self, *input):
        word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_word_text, mask, training = input[:9]
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, ), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        feature_out = self.network_out_features(*input)
        ## feature_out: (batch_size, seq_len, hidden_size)
    
        ## mask padding elements
        seq_len= mask.size(1)
        hidden_size = feature_out.size(2)
        if type(mask) != type(None):
            mask = mask.view(batch_size, seq_len, 1).float()
            feature_out = feature_out*mask
        if self.words2sent == "ATTENTION" or self.words2sent == "ATT":
            feature_out = feature_out.view(batch_size*seq_len, hidden_size).contiguous()
            weights = torch.exp(self.word_weights(feature_out)).view(batch_size, seq_len, 1).contiguous()
            masked_weights = weights*mask
            masked_sums = masked_weights.sum(1, keepdim=True)
            norm_weights = masked_weights/masked_sums
            sent_out = feature_out.view(batch_size, seq_len, hidden_size)*norm_weights
            sent_out = sent_out.sum(1)
        elif self.words2sent == "MAXPOOLING" or self.words2sent == "MAX":
            feature_out = feature_out.transpose(2,1).contiguous()
            sent_out = F.max_pool1d(feature_out, feature_out.size(2)).view(batch_size, -1)
        elif self.words2sent == "MINPOOLING" or self.words2sent == "MIN":
            feature_out = feature_out.transpose(2,1).contiguous()
            sent_out = F.min_pool1d(feature_out, feature_out.size(2)).view(batch_size, -1)
        elif self.words2sent == "AVGPOOLING" or self.words2sent == "AVG":
            feature_out = feature_out.transpose(2,1).contiguous()
            sent_out = F.avg_pool1d(feature_out, feature_out.size(2)).view(batch_size, -1)
        else:
            print("ERROR, word2sent only permit ATTENTION/MAXPOOLING/MINPOOLING/AVGPOOLING, current input: ", self.words2sent)
            exit(0)
        
        ## merge sent represent with features
        feature_list = [sent_out]
        for idx in range(self.feature_num):
            feature_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        final_feature = torch.cat(feature_list, 1)
        outputs = self.hidden2tag(self.dropout(final_feature))
        ## outputs: (batch_size, label_alphabet_size)
        if self.words2sent == "ATTENTION" or self.words2sent == "ATT":
            return outputs, norm_weights.squeeze(2)
        else:
            return outputs, None
