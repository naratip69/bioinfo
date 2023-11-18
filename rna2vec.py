# from Bio import SeqIO
import sys
import gensim.models.keyedvectors as word2vec
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle
import pdb
import tensorflow as tf
# from tensorflow import keras
# from dna2vec.dna2vec.multi_k_model import MultiKModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from gcgc import KmerTokenizer
# from protein2vec import proteinTokenizer,translate,read_fa
from sklearn.model_selection import train_test_split,KFold
from keras.layers import Input, Concatenate, Conv1D,MaxPool1D,Embedding,Lambda,Dense,Flatten,Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,balanced_accuracy_score,matthews_corrcoef,classification_report,confusion_matrix
# import esm
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import h5py

def get_embed_dim(embed_file):
    with open(embed_file,'rb') as f:
        pepEmbedding = pickle.load(f,encoding='latin1')
        
    embedded_dim = pepEmbedding[0].shape
    # print(pd.DataFrame(pepEmbedding[0]).head())
    # print(embedded_dim)
    n_aa_symbols, embedded_dim = embedded_dim
    # print(n_aa_symbols, embedded_dim)
    # = embedded_dim[0]
    embedding_weights = np.zeros((n_aa_symbols + 1,embedded_dim))
    embedding_weights[1:,:] = pepEmbedding[0]
    
    return embedded_dim, embedding_weights, n_aa_symbols

def get_6_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**6
    for i in range(0,end):
        n=i
        ch0=chars[int(n%base)]
        n=n/base
        ch1=chars[int(n%base)]
        n=n/base
        ch2=chars[int(n%base)]
        n=n/base
        ch3=chars[int(n%base)]
        n=n/base
        ch4=chars[int(n%base)]
        n=n/base
        ch5=chars[int(n%base)]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    return  nucle_com
def get_6_nucleotide_composition(tris, seq, ordict):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    #tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer)
            tri_feature.append(ordict[str(ind)])
        else:
            tri_feature.append(-1)
    #tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()        
    return np.asarray(tri_feature)

def read_rna_dict(file):
    odr_dict = {}
    with open(file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict

def make_data_set():
    rnaDict = read_fa('./data/RPI1807_RNA_seq.fa')
    proteinDict = read_fa('./data/RPI1807_protein_seq.fa')

    pd_neg = pd.read_csv('./data/RPI1807_NegativePairs.txt',delimiter='\t',header=None)
    pd_pos = pd.read_csv('./data/RPI1807_PositivePairs.txt',delimiter='\t',header=None)

    rnaList = []
    proteinList = []
    classList = []
    protein_maxlen,rna_maxlen = -1,-1
    for index,row in pd_neg.iterrows():
        rna = rnaDict.get(row[1])
        protein = proteinDict.get(row[0])
        rnaList.append(rna)
        proteinList.append(protein)
        classList.append('NegativePair')
        if(len(protein) > protein_maxlen):
            protein_maxlen = len(protein)
        if(len(rna) > rna_maxlen):
            rna_maxlen = len(rna)
    for index,row in pd_pos.iterrows():
        # print(row)
        rna = rnaDict.get(row[1])
        protein = proteinDict.get(row[0])
        rnaList.append(rna)
        proteinList.append(protein)
        classList.append('PositivePair')
        if(len(protein) > protein_maxlen):
            protein_maxlen = len(protein)
        if(len(rna) > rna_maxlen):
            rna_maxlen = len(rna)

    data_set = pd.DataFrame(list(zip(proteinList,rnaList,classList)), columns=['protein','rna','lpis'])
    print(rna_maxlen,protein_maxlen) # 3396 1733
    data_set.to_csv('./data/data_set.csv',index=False)

def padding_sequence(seq, max_len = 2695, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq
def rnaTokenizer(rna):
    rna_array = []
    trids = get_6_trids()
    nn_dict = read_rna_dict('./data/rna_dict.txt')
    for rna_seq in rna:
        rna_seq_pad = padding_sequence(rna_seq,max_len=501,repkey='N')
        tri_feature = get_6_nucleotide_composition(trids,rna_seq_pad,nn_dict) 
        rna_array.append(tri_feature)
    return np.array(rna_array)
# def protein_to_sentence(protiens):
#     protein_array = []
#     maxlen = 472
#     for pt in protiens:
#         pt_tran = translate(pt)
#         pt_sen = proteinTokenizer(pt_tran)
#         protein_array.append(pt_sen)
#     protein_array = [[0]*(maxlen - len(pt)) + pt for pt in protein_array]
#     return np.array(protein_array)
def matrix_mul(ip):
    return tf.linalg.matmul(ip[0],ip[1])


def get_model():
    protein2vec = Word2Vec.load('./data/proteinModel.model')
    embedded_rna_dim, embedding_rna_weights, n_nucl_symbols = get_embed_dim('./data/rnaEmbedding25.pickle')
    protein_embed = protein2vec.wv.vectors
    # print(protein_embed[:5])
    # print(embedding_rna_weights.shape,protein_embed.shape) #(4097,25) (248,343)
    # print(embedded_rna_dim,n_nucl_symbols) #25 4096
    rna_input = Input((496,),name='rna_input')
    protein_input = Input((2048,1,),name='protein_input')
    rnaEmbed = Embedding(input_dim=4097,output_dim=25,weights=[embedding_rna_weights],trainable=True)(rna_input)
    # proteinEmbed = Embedding(input_dim=248,output_dim=343,trainable=True)(protein_input)
    conv_pro = Conv1D(filters=32,use_bias=True,kernel_size=4)(protein_input)
    conv_rna = Conv1D(filters=32,use_bias=True,kernel_size=4)(rnaEmbed)
    pro_pool = MaxPool1D(pool_size=48,strides=4,padding='valid')(conv_pro)
    rna_pool = MaxPool1D(pool_size=50,strides=2,padding='valid')(conv_rna)
    conv_pro2 = Conv1D(filters=49,use_bias=True,kernel_size=4)(pro_pool)
    conv_rna2 = Conv1D(filters=49,use_bias=True,kernel_size=4)(rna_pool)
    pro_pool2 = MaxPool1D(pool_size=32,strides=4)(conv_pro2)
    rna_pool2 = MaxPool1D(pool_size=24,strides=4)(conv_rna2)
    feature_fusion = Lambda(matrix_mul)((pro_pool2,rna_pool2)) # (?,48,128) , (?,48,128)
    flat = Flatten()(feature_fusion)
    fully_conect = Dense(64,activation='relu',use_bias=True)(flat)
    m1 = Dropout(0.5)(fully_conect)
    m2 = Dense(32,activation='relu',use_bias=True)(m1)
    m3 = Dropout(0.5)(m2)
    m5 = Dense(1,activation='softplus')(m3)
    output = Dense(1,activation='sigmoid',name='output')(m5)
    model = Model(inputs=[rna_input,protein_input],outputs=output)
    # print(model.summary())
    return model


data_set = pd.read_csv('./data/data_set.csv')
# print(data_set.head())
X = data_set.drop('lpis',axis=1).values
y = data_set['lpis'].values
y = [int(e=='PositivePair') for e in y]
y = np.array(y)
y.astype('float32')
y.reshape(-1)
print(y.shape)
#print(X.shape) # (3243, 2)
#print(y.shape) # (3243, )
# maxlen = -1
# for row in X:
#     tri_fea = get_4_nucleotide_composition(trids, row[1])
#     if len(tri_fea) > maxlen:
#         maxlen = len(tri_fea)
# print(maxlen) #3391

# print(rna_array.shape,protein_array.shape) # (3243,496) (3243,1415)
train_X,test_X,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
rna_train = rnaTokenizer(train_X[:,1])
protein_train = (train_X[:,0])
rna_test = rnaTokenizer(test_X[:,1])
protein_test = (test_X[:,0])
print(test_X.shape,y_test.shape)
print(f'negative: {np.sum(y_train==0)} , positive : {np.sum(y_train)}')
print(torch.cuda.is_available())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.is_available():
# torch.set_default_tensor_type(torch.cuda.HalfTensor)
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
pretrain_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
pretrain_model.full() if device=='cpu' else pretrain_model.half()
ids_train = tokenizer(list(protein_train), add_special_tokens=True, padding="longest")
ids_test = tokenizer(list(protein_test), add_special_tokens=True, padding="longest")

input_ids_train = torch.tensor(ids_train['input_ids']).to(device)
input_ids_test = torch.tensor(ids_test['input_ids']).to(device)
attention_mask_train = torch.tensor(ids_train['attention_mask']).to(device)
attention_mask_test = torch.tensor(ids_test['attention_mask']).to(device)
with torch.no_grad():
    embedding_repr_train = pretrain_model(input_ids=input_ids_train, attention_mask=attention_mask_train)
    embedding_repr_test = pretrain_model(input_ids=input_ids_test, attention_mask=attention_mask_test)
# print(embedding_repr.last_hidden_state.shape) #torch.Size([1622, 2, 1024])
emb_train = embedding_repr_train.last_hidden_state.cpu().numpy()
emb_test = embedding_repr_test.last_hidden_state.cpu().numpy()
emb_train = emb_train.reshape((emb_train.shape[0],-1))
emb_test = emb_test.reshape((emb_test.shape[0],-1))
# emb = emb.reshape((emb.shape[0],-1))
# print(emb.shape) #(1622, 2048)
model = get_model()
print(model.summary())
adam = Adam(learning_rate=0.00001)
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])



model.fit([rna_train,emb_train],y_train,batch_size=32,epochs=100,verbose=1)

model.save('models/model.h5')

y_pred = model.predict([rna_test,emb_test])
y_pred.reshape(-1)
y_pred = y_pred > 0.5
y_pred.astype('float32')
print(y_pred.shape,y_test.shape)
print(y_pred[:5])
print(y_test[:5])
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
acc1 = accuracy_score(y_test,y_pred) 
print(acc1)
# cv = KFold(n_splits=2,shuffle=True,random_state=42)
# fold_no = 1
# metrics_per_fold = []
# show = False
# for train, test in cv.split(X,y):
#     print("     ")
#     print(f'Training for fold {fold_no}')

#     # scaler = MinMaxScaler()
#     train_X = X[train]
#     test_X = X[test]
#     rna_train = rnaTokenizer(train_X[:,1])
#     protein_train = (train_X[:,0])
#     rna_test = rnaTokenizer(test_X[:,1])
#     protein_test = (test_X[:,0])
#     print(test_X.shape,y[test].shape)
#     print(f'negative: {np.sum(y[test]==0)} , positive : {np.sum(y[test])}')
#     print(torch.cuda.is_available())

#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     # if torch.cuda.is_available():
#     #     torch.set_default_tensor_type(torch.cuda.HalfTensor)
#     tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# # Load the model
#     pretrain_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
#     pretrain_model.full() if device=='cpu' else pretrain_model.half()
#     ids_train = tokenizer(list(protein_train), add_special_tokens=True, padding="longest")
#     ids_test = tokenizer(list(protein_test), add_special_tokens=True, padding="longest")

#     input_ids_train = torch.tensor(ids_train['input_ids']).to(device)
#     input_ids_test = torch.tensor(ids_test['input_ids']).to(device)
#     attention_mask_train = torch.tensor(ids_train['attention_mask']).to(device)
#     attention_mask_test = torch.tensor(ids_test['attention_mask']).to(device)
#     with torch.no_grad():
#         embedding_repr_train = pretrain_model(input_ids=input_ids_train, attention_mask=attention_mask_train)
#         embedding_repr_test = pretrain_model(input_ids=input_ids_test, attention_mask=attention_mask_test)
#     # print(embedding_repr.last_hidden_state.shape) #torch.Size([1622, 2, 1024])
#     emb_train = embedding_repr_train.last_hidden_state.cpu().numpy()
#     emb_test = embedding_repr_test.last_hidden_state.cpu().numpy()
#     emb_train = emb_train.reshape((emb_train.shape[0],-1))
#     emb_test = emb_test.reshape((emb_test.shape[0],-1))
#     # emb = emb.reshape((emb.shape[0],-1))
#     # print(emb.shape) #(1622, 2048)
#     model = get_model()
#     model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#     if(not show):
#         print(model.summary())
#         show = True

#     history = model.fit([rna_train,emb_train],y[train],batch_size=32,epochs=32,verbose=1)

#     model.save('models/model_fold_'+str(fold_no)+".h5")
    
#     _,score = model.evaluate([rna_test,emb_test],y[test],verbose=0)
#     metrics_per_fold.append(score)
#     fold_no = fold_no + 1

# for score in metrics_per_fold:
#     print(score)