from Bio import SeqIO
import sys
import gensim.models.keyedvectors as word2vec
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle
import pdb
import tensorflow as tf
# from dna2vec.dna2vec.multi_k_model import MultiKModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gcgc import KmerTokenizer
import re

def translate(seq):
    seq = re.sub(r'(A|G|V)+','a',seq)
    # print(seq)
    seq = re.sub(r'(I|L|F|P)+','b',seq)
    seq = re.sub(r'(Y|M|T|S)+','c',seq)
    seq = re.sub(r'(H|N|Q|W)+','d',seq)
    seq = re.sub(r'(R|K)+','e',seq)
    seq = re.sub(r'(D|E)+','f',seq)
    seq = re.sub(r'C+','g',seq)
    return seq

def proteinTokenizer(seq):
    out = []
    char2num = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6}
    for i in range(len(seq)):
        if(i+2 >= len(seq) or (not ((seq[i] in char2num) and (seq[i+1] in char2num) and (seq[i+2] in char2num)))):
            out.append(343)
        else:
            out.append(( char2num[seq[i+2]]*49 +char2num[seq[i+1]]*7+ char2num[seq[i]] ))
    return out

def read_fa(file):
    fasta_sequences = SeqIO.parse(open(file),'fasta')
    fastaDict = {}
    for fasta in fasta_sequences:
        name, seq = fasta.id , str(fasta.seq)
        fastaDict[name] = seq
    return fastaDict

def protein_to_sequences():
    file = './data/RPI1807_protein_seq.fa'
    fastaDict = read_fa(file)
    maxlen = -1
    sentences = []
    for key,val in fastaDict.items():
        # print(val)
        seq = translate(val)
        sentences.append(proteinTokenizer(seq))
        if(len(seq) > maxlen):
            maxlen = len(seq)
    sentences = [['0']*(maxlen - len(seq)) + seq for seq in sentences]
    return sentences

def train_protein():
    sentences = protein_to_sequences()
    model = Word2Vec(sentences=sentences,epochs = 10,batch_words=100,vector_size=343)
    # print(len(sentences[0])) #1415
    # test_seq = 'MSWDVIKHPHVTEKAMNDMDFQNKLQFAVDDRASKGEVADAVEEQYDVTVEQVNTQNTMDGEKKAVVRLSEDDDAQEVASRIGVF'
    # test_seq  = translate(test_seq)
    model.save('./data/proteinModel.model')


train_protein()



