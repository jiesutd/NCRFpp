# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-02-28 23:09:54
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from charbilstm import CharBiLSTM
from charbigru import CharBiGRU
from charcnn import CharCNN

class CNN(nn.Module):
    def __init__(self, data):
        super(CNN, self).__init__()
        print "build batched cnn..."
        self.gpu = data.HP_gpu
        self.use_char = data.HP_use_char
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.average_batch = data.HP_average_batch_loss
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_features == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_features == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_features == "GRU":
                self.char_feature = CharBiGRU(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print "Error char feature selection, please check parameter data.char_features (either CNN or LSTM)."
                exit(0)
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
        
        # The CNN takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        cnn_hidden = data.HP_hidden_dim
        self.word2cnn = nn.Linear(self.embedding_dim + self.char_hidden_dim, cnn_hidden)
        self.cnn_layer = 2
        print "CNN layer: ", self.cnn_layer
        self.cnn_list = nn.ModuleList()
        self.cnn_drop_list = nn.ModuleList()
        self.cnn_batchnorm_list = nn.ModuleList()
        kernel = 3
        pad_size = (kernel-1)/2
        for idx in range(self.cnn_layer):
            self.cnn_list.append(nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=kernel, padding=pad_size))
            self.cnn_drop_list.append(nn.Dropout(data.HP_dropout))
            self.cnn_batchnorm_list.append(nn.BatchNorm1d(cnn_hidden))
        

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(cnn_hidden, data.label_alphabet_size)

        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.word2cnn = self.word2cnn.cuda()
            for idx in range(self.cnn_layer):
                self.cnn_list[idx] = self.cnn_list[idx].cuda()
                self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            self.hidden2tag = self.hidden2tag.cuda()


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def get_cnn_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs =  self.word_embeddings(word_inputs)

        if self.use_char:
            ## calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            ## concat word and char together
            word_embs = torch.cat([word_embs, char_features], 2)
        word_embs = self.drop(word_embs)
        word_in = F.tanh(self.word2cnn(word_embs)).transpose(2,1).contiguous()
        for idx in range(self.cnn_layer):
            if idx == 0:
                cnn_feature = F.relu(self.cnn_list[idx](word_in))
            else:
                cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
            cnn_feature = self.cnn_drop_list[idx](cnn_feature)
            cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)

        cnn_out = cnn_feature.transpose(2,1).contiguous()
        # cnn_out (batch_size, seq_len, hidden_size)
        return cnn_out


    def get_output_score(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        cnn_out = self.get_cnn_features(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        outputs = self.hidden2tag(cnn_out)
        return outputs
    

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        ## mask is not used
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        total_word = batch_size * seq_len
        
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outs = self.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # outs (batch_size, seq_len, label_vocab)
        outs = outs.view(total_word, -1)
        score = F.log_softmax(outs, 1)
        loss = loss_function(score, batch_label.view(total_word))
        if self.average_batch:
            loss = loss / batch_size
        _, tag_seq  = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)

        return loss, tag_seq




    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        outs = self.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        outs = outs.view(total_word, -1)
        _, tag_seq  = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        ## filter padded position with zero
        decode_seq = mask.long() * tag_seq
        return decode_seq

        