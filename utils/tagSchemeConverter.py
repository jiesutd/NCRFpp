# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-11-27 16:53:36
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-01-09 21:39:10


"""
    convert NER/Chunking tag schemes, i.e. BIO->BIOES, BIOES->BIO, IOB->BIO, IOB->BIOES
"""
from __future__ import print_function

import sys


def BIO2BIOES(input_file, output_file):
    print("Convert BIO -> BIOES for file:", input_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in labels[idx]:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    label_type = labels[idx].split('-')[-1]
                    if "B-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" S-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" B-"+label_type+"\n")
                    elif "I-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" E-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" I-"+label_type+"\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIOES file generated:", output_file)



def BIOES2BIO(input_file, output_file):
    print("Convert BIOES -> BIO for file:", input_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in labels[idx]:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    label_type = labels[idx].split('-')[-1]
                    if "E-" in labels[idx]:
                        fout.write(words[idx]+" I-"+label_type+"\n")
                    elif "S-" in labels[idx]:
                        fout.write(words[idx]+" B-"+label_type+"\n")
                    else:
                        fout.write(words[idx]+" "+labels[idx]+"\n")     
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIO file generated:", output_file)


def IOB2BIO(input_file, output_file):
    print("Convert IOB -> BIO for file:", input_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "I-" in labels[idx]:
                    label_type = labels[idx].split('-')[-1]
                    if (idx == 0) or (labels[idx-1] == "O") or (label_type != labels[idx-1].split('-')[-1]):
                        fout.write(words[idx]+" B-"+label_type+"\n")
                    else:
                        fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIO file generated:", output_file)


def choose_label(input_file, output_file):
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    with open(output_file,'w') as fout:
        for line in fins:
            if len(line) < 3:
                fout.write(line)
            else:
                pairs = line.strip('\n').split(' ')
                fout.write(pairs[0]+" "+ pairs[-1]+"\n")


if __name__ == '__main__':
    '''Convert NER tag schemes among IOB/BIO/BIOES.
        For example: if you want to convert the IOB tag scheme to BIO, then you run as following:
            python tagSchemeConverter.py IOB2BIO input_iob_file output_bio_file
        Input data format is the standard CoNLL 2003 data format.
    '''
    if sys.argv[1].upper() == "IOB2BIO":
        IOB2BIO(sys.argv[2],sys.argv[3])
    elif sys.argv[1].upper() == "BIO2BIOES":
        BIO2BIOES(sys.argv[2],sys.argv[3])
    elif sys.argv[1].upper() == "BIOES2BIO":
        BIOES2BIO(sys.argv[2],sys.argv[3])
    elif sys.argv[1].upper() == "IOB2BIOES":
        IOB2BIO(sys.argv[2],"temp")
        BIO2BIOES("temp",sys.argv[3])
    else:
        print("Argument error: sys.argv[1] should belongs to \"IOB2BIO/BIO2BIOES/BIOES2BIO/IOB2BIOES\"")
