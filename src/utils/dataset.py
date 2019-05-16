import re
import torch
import string
import pickle
import os,sys
import random
import functools 
import numpy as np
import pandas as pd
import sentencepiece as spm
from string import punctuation
import torch.utils.data as data
from sklearn.model_selection import train_test_split


def pckl(obj,path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def upkl(path):
    with open(path, 'rb') as handle:
        _ = pickle.load(handle)
    return _  


def plain_text_split(text):
    return text.split(' ')


class Tokenizer(object):
    def __init__(self,
                 model_path):
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        self.sp = sp
        
    def enc_text2seq(self,text):
        return self.sp.EncodeAsIds(text)

    def enc_text2pieces(self,text):
        return self.sp.EncodeAsPieces(text)

    def dec_pieces2text(self,piece_list):
        return self.sp.DecodePieces(piece_list) 

    
class EmotionDataset(data.Dataset):
    def __init__(self,
                 
                 mode='train',
                 max_input_len = 100,
                 holdout_share=0.25,
                 
                 df_path='../data/fin_tweets_train.feather',
                 language='en',
                 random_state=228,
                 
                 embedding_type='dict',
                 tokenizer_path = '',
                 
                 pad_token_id=0,
                 w2i=None
                ):
        
        self.mode = mode
        df = pd.read_feather(df_path)
        df = df[df.lang==language]
        self.holdout_share = holdout_share

        self.texts = list(df.processed.values)
        self.targets = list(df.target.values)
        self.idx = list(range(0,len(self.texts)))
        
        self.random_state = random_state
        self.embedding_type = embedding_type
        self.train_idx, self.val_idx = train_test_split(self.idx,
                                                        test_size=self.holdout_share,
                                                        random_state=self.random_state)
        if self.embedding_type=='bpe':
            self.tokenizer = Tokenizer(tokenizer_path)
        self.w2i = w2i
        self.pad_token_id = pad_token_id
        self.max_input_len = max_input_len
        
    def __len__(self):
        if self.mode=='train':
            return len(self.train_idx)
        elif self.mode=='val':
            return len(self.val_idx)


    @staticmethod
    def pad_sequence(seq,
                     max_len,
                     pad_token=0):
        return np.asarray(list(seq) + [pad_token] * (max_len - len(seq)))        

    
    def __getitem__(self, idx):
        if self.mode=='train':
            idx = self.train_idx[idx]
        elif self.mode=='val':
            idx = self.val_idx[idx]
        
        text = self.texts[idx]
        print(text)
        target = self.targets[idx]
            
        if self.embedding_type=='dict':
            text_list = text.split(' ')
            seq_len = min(len(text_list),self.max_input_len)
            token_list = [self.w2i[_] for _ in text_list]
            return_object = self.pad_sequence(token_list,
                                              self.max_input_len,
                                              self.pad_token_id)[:self.max_input_len]
            
        elif self.embedding_type=='bpe':
            token_list = self.tokenizer.enc_text2seq(text)
            seq_len = min(len(token_list),self.max_input_len)
            return_object = self.pad_sequence(token_list,
                                              self.max_input_len,
                                              self.pad_token_id)[:self.max_input_len]                        
        
        elif self.embedding_type=='bag' or self.embedding_type=='wbag':
            text_list = text.split(' ')
            seq_len = min(len(text_list),self.max_input_len)
            text_list.extend([' '] * (self.max_input_len - len(text_list)))
            return_object = text_list[:self.max_input_len]

        # in case of dictionaries / bpe we just return padded sequences
        # in case of embedding bags - we return text numpy arrays
        return_tuple = (
            return_object,
            target,
            seq_len
        )
        return return_tuple