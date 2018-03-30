# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-09 22:00:36

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cnn import CNN
from crf import CRF

class CNN_CRF(nn.Module):
    def __init__(self, data):
        super(CNN_CRF, self).__init__()
        print "build batched cnncrf..."
        self.gpu = data.HP_gpu
        self.average_batch = data.HP_average_batch_loss
        ## add two more label for downlayer cnn, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.cnn = CNN(data)
        self.crf = CRF(label_size, self.gpu)


    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        outs = self.cnn.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        vector_dim = outs.size(2)
        new_mask = mask.unsqueeze(2).expand(batch_size, seq_len, vector_dim).float()
        outs = new_mask * outs
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        # word_num = mask.sum()
        # total_loss = total_loss / word_num.data[0]
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq


    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        outs = self.cnn.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return tag_seq


    def get_cnn_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        return self.cnn.get_cnn_features(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        