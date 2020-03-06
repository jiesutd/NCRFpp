# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-01 21:11:50
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-29 14:27:55

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
import numpy as np

class SentClassifier(nn.Module):
    def __init__(self, data):
        super(SentClassifier, self).__init__()
        if not data.silence:
            print("build sentence classification network...")
            print("use_char: ", data.use_char)
            if data.use_char:
                print("char feature extractor: ", data.char_feature_extractor)
            print("word feature extractor: ", data.word_feature_extractor)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        label_size = data.label_alphabet_size
        self.word_hidden = WordSequence(data)



    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        outs, _ = self.word_hidden.sentence_representation(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,mask)
        batch_size = word_inputs.size(0)
        # loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        outs = outs.view(batch_size, -1)
        # print("a",outs)
        # score = F.log_softmax(outs, 1)
        # print(score.size(), batch_label.view(batch_size).size())
        # print(score)
        # print(batch_label)
        # exit(0)
        # print(batch_label)
        # print(outs.size(), batch_label.size())
        total_loss = F.cross_entropy(outs, batch_label.view(batch_size))
        # total_loss = loss_function(score, batch_label.view(batch_size))
        
        _, tag_seq  = torch.max(outs, 1)
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        outs,_ = self.word_hidden.sentence_representation(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,mask)
        batch_size = word_inputs.size(0)
        outs = outs.view(batch_size, -1)
        _, tag_seq  = torch.max(outs, 1)
        # if a == 0:
        #     print(tag_seq)
        return tag_seq

    def get_target_probability(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        outs, weights = self.word_hidden.sentence_representation(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,mask)
        batch_size = word_inputs.size(0)
        outs = outs.view(batch_size, -1)
        _, tag_seq  = torch.max(outs, 1)
        outs = outs[:,1:]
        sf = nn.Softmax(1)
        prob_outs = sf(outs)
        if self.gpu:
            prob_outs = prob_outs.cpu()
            if type(weights) != type(None):
                weights = weights.cpu()
    
        if type(weights) != type(None):
            weight = weights.detach().numpy()

        probs = np.insert(prob_outs.detach().numpy(), 0, 0, axis=1)
        
        return probs, weights


