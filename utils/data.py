# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2017-11-30 11:30:13
import sys
import numpy as np
from alphabet import Alphabet
from functions import *
import cPickle as pickle


START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"

class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 512
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.word_alphabet.add(START)
        self.word_alphabet.add(UNKNOWN)
        self.char_alphabet.add(START)
        self.char_alphabet.add(UNKNOWN)
        self.char_alphabet.add(PADDING)
        self.label_alphabet = Alphabet('label')
        self.tagScheme = "NoSeg"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.word_emb_dim = 50
        self.char_emb_dim = 30
        self.pretrain_word_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = True
        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0
        self.HP_clip = 5.0
        self.HP_momentum = 0

        
    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("     Hyperpara  iteration: %s"%(self.HP_iteration))
        print("     Hyperpara  batch size: %s"%(self.HP_batch_size))
        print("     Hyperpara          lr: %s"%(self.HP_lr))
        print("     Hyperpara    lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s"%(self.HP_clip))
        print("     Hyperpara    momentum: %s"%(self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s"%(self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s"%(self.HP_bilstm))
        print("     Hyperpara    use_char: %s"%(self.HP_use_char))
        print("     Hyperpara         GPU: %s"%(self.HP_gpu))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file,'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s"%(old_size, self.label_alphabet_size))


    def extend_word_char_alphabet(self, input_file_list):
        old_word_size = self.word_alphabet_size
        old_char_size = self.char_alphabet_size
        for input_file in input_file_list:
            in_lines = open(input_file,'r').readlines()
            for line in in_lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    self.word_alphabet.add(word)
                    for char in word:
                        self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        print("Extend word/char alphabet finished!")
        print("     old word:%s -> new word:%s"%(old_word_size, self.word_alphabet_size))
        print("     old char:%s -> new char:%s"%(old_char_size, self.char_alphabet_size))
        for input_file in input_file_list:
            print("     from file:%s"%(input_file))



    def build_alphabet(self, input_file):
        in_lines = open(input_file,'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0].decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()       


    def build_word_pretrain_emb(self, emb_path, norm=False):
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim, norm)


    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_WORD_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_WORD_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_WORD_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_WORD_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    # def generate_instance_variable(self):
    #     print "Generate variable for data."
    #     import torch
    #     import torch.autograd as autograd
    #     for words, chars, label in self.train_Ids:
    #         words = autograd.Variable(torch.LongTensor(words))
    #         new_chars = []
    #         for char in chars:
    #             new_chars.append(autograd.Variable(torch.LongTensor(char)))
    #         label = autograd.Variable(torch.LongTensor(label))
    #         self.train_Vars.append([words, new_chars, label])
    #     for words, chars, label in self.dev_Ids:
    #         words = autograd.Variable(torch.LongTensor(words), volatile = True)
    #         new_chars = []
    #         for char in chars:
    #             new_chars.append(autograd.Variable(torch.LongTensor(char), volatile = True))
    #         label = autograd.Variable(torch.LongTensor(label), volatile = True)
    #         self.dev_Vars.append([words, new_chars, label])
    #     for words, chars, label in self.test_Ids:
    #         words = autograd.Variable(torch.LongTensor(words), volatile = True)
    #         new_chars = []
    #         for char in chars:
    #             new_chars.append(autograd.Variable(torch.LongTensor(char), volatile = True))
    #         label = autograd.Variable(torch.LongTensor(label), volatile = True)
    #         self.test_Vars.append([words, new_chars, label])
    #     for words, chars, label in self.raw_Ids:
    #         words = autograd.Variable(torch.LongTensor(words), volatile = True)
    #         new_chars = []
    #         for char in chars:
    #             new_chars.append(autograd.Variable(torch.LongTensor(char), volatile = True))
    #         label = autograd.Variable(torch.LongTensor(label), volatile = True)
    #         self.raw_Vars.append([words, new_chars, label])
    #     return True


    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file,'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy] + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, output_file))





