# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import pandas as pd

nlp = spacy.load('en_core_web_sm')


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    return matrix

def process(filename):
    # fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # lines = fin.readlines()
    # fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    fin=pd.read_csv(filename,sep='\t',names=['label','text'])
    for i in range(len(fin['label'])):
        # text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        # text ,label= [s.lower().strip() for s in lines[i].partition("\t")]
        text= fin['text'][i].lower()
        # aspect = lines[i + 1].lower().strip()
        # adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        adj_matrix = dependency_adj_matrix(text)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)        
    fout.close() 

if __name__ == '__main__':
    process('./datasets/TREC/train.csv')
    process('./datasets/TREC/test.csv')
    # process('./datasets/semeval14/restaurant_train.raw')
    # process('./datasets/semeval14/restaurant_test.raw')
    # process('./datasets/semeval14/laptop_train.raw')
    # process('./datasets/semeval14/laptop_test.raw')
    # process('./datasets/semeval15/restaurant_train.raw')
    # process('./datasets/semeval15/restaurant_test.raw')
    # process('./datasets/semeval16/restaurant_train.raw')
    # process('./datasets/semeval16/restaurant_test.raw')