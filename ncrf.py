# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-29 14:31:08

from __future__ import print_function
import time
import sys
import os
import argparse
import random
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.seqlabel import SeqLabel
from model.sentclassifier import SentClassifier
from utils.data import Data
from main import *
from tqdm import tqdm

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

class NCRF:
    def __init__(self):
        print("Python Version: %s.%s"%(sys.version_info[0],sys.version_info[1]))
        print("PyTorch Version:%s"%(torch.__version__))
        print("Process ID: ", os.getpid())
        self.data = Data()
        self.data.HP_gpu = torch.cuda.is_available()
        if self.data.HP_gpu:
            self.data.device = 'cuda'
        print("GPU:", self.data.HP_gpu, "; device:", self.data.device)
        self.optimizer = None
        self.model = None
        

    def read_data_config_file(self, config_dir):
        self.data.read_config(config_dir)


    def manual_data_setting(self, setting_dict):
        ## set data through manual dict, all value should be in string format.
        self.data.manual_config(setting_dict)


    def initialize_model_and_optimizer(self):
        if self.data.sentence_classification:
            self.model = SentClassifier(self.data)
        else:
            self.model = SeqLabel(self.data) 
        if self.data.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.data.HP_lr, momentum=self.data.HP_momentum,weight_decay=self.data.HP_l2)
        elif self.data.optimizer.lower() == "adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.data.HP_lr, weight_decay=self.data.HP_l2)
        elif self.data.optimizer.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.data.HP_lr, weight_decay=self.data.HP_l2)
        elif self.data.optimizer.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.data.HP_lr, weight_decay=self.data.HP_l2)
        elif self.data.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.data.HP_lr, weight_decay=self.data.HP_l2)
        else:
            print("Optimizer illegal: %s"%(self.data.optimizer))
            exit(1)


    def initialize_data(self, input_list=None):
        self.data.initial_alphabets(input_list)
        if self.data.use_word_emb and self.data.use_word_seq:
            self.data.build_pretrain_emb()


    def initialization(self, input_list=None):
        ## must initialize data before initialize model and optimizer, as alphabet size and pretrain emb matters
        '''
        input_list: [train_list, dev_list, test_list]
              train_list/dev_list/test_list: [sent_list, label_list, feature_list]
                      sent_list: list of list [[word1, word2,...],...,[wordx, wordy]...]
                      label_list:     if sentence_classification: 
                                           list of labels [label1, label2,...labelx, labely,...]
                                      else: 
                                           list of list [[label1, label2,...],...,[labelx, labely,...]]
                      feature_list:   if sentence_classification: 
                                           list of labels [[feat1, feat2,..],...,[feat1, feat2,..]], len(feature_list)= sentence_num
                                      else: 
                                           list of list [[[feat1, feat2,..],...,[feat1, feat2,..]],...,[[feat1, feat2,..],...,[feat1, feat2,..]]], , len(feature_list)= sentence_num
        '''
        self.initialize_data(input_list)
        self.initialize_model_and_optimizer()

    def self_generate_instances(self):
        self.data.generate_instance('train')
        self.data.generate_instance('dev')
        self.data.generate_instance('test')


    def generate_instances_from_list(self, input_list, name):
        return self.data.generate_instance_from_list(input_list, name)

    

    def save(self, model_dir = "ncrf.model"):
        print("Save model to file: ", model_dir)
        the_dict = {
        'data':self.data, 
        'state_dict':self.model.state_dict(),
        'optimizer':self.optimizer.state_dict()
        }
        torch.save(the_dict, model_dir)


    def load(self, model_dir = "ncrf.model"):
        the_dict = torch.load(model_dir)
        self.data = the_dict['data']
        self.data.silence = True
        ## initialize the model and optimizer befor load state dict
        self.initialize_model_and_optimizer()
        self.model.load_state_dict(the_dict['state_dict'])
        self.optimizer.load_state_dict(the_dict['optimizer'])
        print("Model loaded from file: ", model_dir)


    def train(self, train_Ids=None, save_model_dir=None):
        '''
        train_Ids: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)
        save_model_dir: model name to be saved
        '''
        if train_Ids:
            self.data.train_Ids = train_Ids
        # print(self.data.train_Ids[0])
        # print(len(self.data.train_Ids))
        # exit(0)
        best_dev = -10
        best_model = None
        for idx in range(self.data.HP_iteration):
            epoch_start = time.time()
            temp_start = epoch_start
            print("Epoch: %s/%s" %(idx,self.data.HP_iteration))
            if self.data.optimizer == "SGD":
                self.optimizer = lr_decay(self.optimizer, idx, self.data.HP_lr_decay, self.data.HP_lr)
            instance_count = 0
            sample_id = 0
            sample_loss = 0
            total_loss = 0
            right_token = 0
            whole_token = 0
            random.shuffle(self.data.train_Ids)
            first_list = ", ".join([self.data.word_alphabet.get_instance(a) for a in self.data.train_Ids[0][0]])
            print("Shuffle: first input: [%s]" %(first_list))
            ## set model in train model
            self.model.train()
            
            batch_size = self.data.HP_batch_size
            batch_id = 0
            train_num = len(self.data.train_Ids)
            total_batch = train_num//batch_size+1
            for batch_id in range(total_batch):
                self.optimizer.zero_grad()
                start = batch_id*batch_size
                end = (batch_id+1)*batch_size
                if end >train_num:
                    end = train_num
                instance = self.data.train_Ids[start:end]
                if not instance:
                    continue
                batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_word_text, batch_label, mask  = batchify_with_label(instance, self.data.HP_gpu, True, self.data.sentence_classification)
                instance_count += 1
                loss, tag_seq = self.model.calculate_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_word_text, batch_label, mask)
                right, whole = predict_check(tag_seq, batch_label, mask, self.data.sentence_classification)
                right_token += right
                whole_token += whole
                # print("loss:",loss.item())
                sample_loss += loss.item()
                total_loss += loss.item()
                if end%500 == 0:
                    temp_time = time.time()
                    temp_cost = temp_time - temp_start
                    temp_start = temp_time
                    print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                    if sample_loss > 1e8 or str(sample_loss) == "nan":
                        print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                        exit(1)
                    sys.stdout.flush()
                    sample_loss = 0
                loss.backward()
                self.optimizer.step()
            temp_time = time.time()
            temp_cost = temp_time - temp_start
            print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
            print("totalloss:", total_loss)
            if total_loss > 1e8 or str(total_loss) == "nan":
                print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                exit(1)
            # continue
            speed, acc, p, r, f, _,_ = evaluate(self.data, self.model, "dev")
            dev_finish = time.time()
            dev_cost = dev_finish - epoch_finish
            
            if self.data.seg:
                print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f))
                current_score = f
            else:
                print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))
                current_score = acc
            if current_score > best_dev:
                if self.data.seg:
                    print("Exceed previous best f score:", best_dev)
                else:
                    print("Exceed previous best acc score:", best_dev)
                if save_model_dir == None:
                    model_name = self.data.model_dir + ".model"
                else:
                    model_name = save_model_dir  + ".model"
                self.save(model_name)
                # torch.save(model.state_dict(), model_name)
                best_dev = current_score
                best_model = model_name
            # ## decode test
            speed, acc, p, r, f, test_result,_ = evaluate(self.data, self.model, "test")
            test_finish = time.time()
            test_cost = test_finish - dev_finish
            if self.data.seg:
                print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
            else:
                print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
            gc.collect()
        if best_model != None:
            self.load(best_model)

    # def evaluate(self):


    def decode(self, raw_Ids):
        '''
        raw_Ids: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)
            ## label should be padded in raw input
        '''
        instances = raw_Ids
        ## set model in eval model
        self.model.eval()
        batch_size = self.data.HP_batch_size
        instance_num = len(instances)
        total_batch = instance_num//batch_size+1
        decode_label = []
        for batch_id in tqdm(range(total_batch)):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > instance_num:
                end =  instance_num
            instance = instances[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_word_text, batch_label, mask  = batchify_with_label(instance, self.data.HP_gpu, False, self.data.sentence_classification)
            tag_seq = self.model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_word_text, None, mask)
            tag_seq = tag_seq[batch_wordrecover.cpu()]
            decode_label += tag_seq.cpu().data.numpy().tolist()
        return  decode_label


    def decode_prob(self, raw_Ids):
        '''
        raw_Ids: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)
            ## label should be padded in raw input
        '''
        if not self.data.sentence_classification:
            print("decode probability is only valid in sentence classification task. Exit.")
            exit(0)
        instances = raw_Ids
        target_probability_list = []
        target_result_list = []
        ## set model in eval model
        self.model.eval()
        batch_size = self.data.HP_batch_size
        instance_num = len(instances)
        total_batch = instance_num//batch_size+1
        for batch_id in tqdm(range(total_batch)):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > instance_num:
                end =  instance_num
            instance = instances[start:end]
            if start%1000 == 0:
                print("Decode: ", start)
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_word_text, batch_label, mask  = batchify_with_label(instance, self.data.HP_gpu, False, self.data.sentence_classification)
            target_probability, _ = self.model.get_target_probability(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_word_text, None, mask)
            target_probability = target_probability[batch_wordrecover.cpu()]
            target_probability_list.append(target_probability)
        target_probabilities = np.concatenate(target_probability_list, axis = 0)
        return target_probabilities

    def decode_prob_and_attention_weights(self, raw_Ids):
        '''
        raw_Ids: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)
            ## label should be padded in raw input
        '''
        if not self.data.sentence_classification:
            print("decode probability is only valid in sentence classification task. Exit.")
            exit(0)
        if self.data.words2sent_representation.upper() != "ATTENTION" and self.data.words2sent_representation.upper() != "ATT":
            print("attention weights are only valid in attention model. Current: %s,  Exit."%(self.data.words2sent_representation))
            exit(0)
        instances = raw_Ids
        target_probability_list = []
        sequence_attention_weight_list = []
        ## set model in eval model
        self.model.eval()
        batch_size = self.data.HP_batch_size
        instance_num = len(instances)
        total_batch = instance_num//batch_size+1
        for batch_id in tqdm(range(total_batch)):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > instance_num:
                end =  instance_num
            instance = instances[start:end]
            if start%1000 == 0:
                print("Decode: ", start)
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_word_text, batch_label, mask  = batchify_with_label(instance, self.data.HP_gpu, False, self.data.sentence_classification)
            target_probability, weights = self.model.get_target_probability(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_word_text, None, mask)
            ## target_probability, weights are both numpy
            target_probability = target_probability[batch_wordrecover.cpu()]
            weights = weights[batch_wordrecover.cpu()]
            target_probability_list.append(target_probability)
            sequence_attention_weight_list += weights.tolist()
        target_probabilities = np.concatenate(target_probability_list, axis = 0)
        print(len(sequence_attention_weight_list))
        ## sequence_attention_weight_list: list with different batch size and many padded 0
        return target_probabilities, sequence_attention_weight_list





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config',  help='Configuration File', default='None')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--wordemb',  help='Embedding for words')
    parser.add_argument('--charemb',  help='Embedding for chars')
    parser.add_argument('--savemodel')
    parser.add_argument('--train') 
    parser.add_argument('--dev' )  
    parser.add_argument('--test') 
    parser.add_argument('--seg') 
    parser.add_argument('--raw') 
    parser.add_argument('--loadmodel')
    parser.add_argument('--output') 
    args = parser.parse_args()

    ncrf = NCRF()
    if args.config == 'None':
        ncrf.data.train_dir = args.train 
        ncrf.data.dev_dir = args.dev 
        ncrf.data.test_dir = args.test
        ncrf.data.model_dir = args.savemodel
        ncrf.save_model_dir = args.savemodel
        ncrf.data.word_emb_dir = args.wordemb
        ncrf.data.char_emb_dir = args.charemb
    else:
        ncrf.read_data_config_file(args.config)
        raw_dir = ncrf.data.raw_dir  ## keep the raw_dir in case of being update when loading model.
        decode_dir = ncrf.data.decode_dir
        nbest = ncrf.data.nbest
    if ncrf.data.status == 'train':
        ncrf.initialization()
        ncrf.self_generate_instances()
        ncrf.data.summary()
        ncrf.train()
    elif ncrf.data.status == 'decode':
        dev_finish  = time.time()
        ncrf.load(ncrf.data.load_model_dir)
        ncrf.data.raw_dir = raw_dir
        ncrf.data.decode_dir = decode_dir
        ncrf.data.nbest = nbest
        ncrf.data.generate_instance('raw')
        ncrf.data.summary()
        # ## decode test
        speed, acc, p, r, f, decode_results, pred_scores = evaluate(ncrf.data, ncrf.model, "raw", ncrf.data.nbest)
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if self.data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
        if ncrf.data.nbest >1 and not ncrf.data.sentence_classification:
            ncrf.data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            ncrf.data.write_decoded_results(decode_results, 'raw')
    
        

    

