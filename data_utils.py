# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import re
# import dependency_graph

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue
    return word_vec

def clean_str_sst(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()
def clean_str(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()   
def process(dataset):
    data_dir = "./datasets/" + dataset
    root = os.path.join(data_dir,"rt-polaritydata")
    saved_path=data_dir
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    datas=[]
    for polarity in  ("neg","pos"):
        filename = os.path.join(root,polarity) 
        records=[]
        with open(filename,encoding="utf-8",errors="replace") as f:
            for i,line in enumerate(f):
                records.append({"text":clean_str(line).strip(),"label": 0 if polarity == "pos" else 1})
        datas.append(pd.DataFrame(records))
    df = pd.concat(datas)
    from sklearn.utils import shuffle  
    df = shuffle(df).reset_index()
    split_index = [True] * int (len(df) *0.9) + [False] *(len(df)-int (len(df) *0.9))
    train = df[split_index]
    dev = df[~np.array(split_index)]
    train_filename=os.path.join(saved_path,"train.csv")
    test_filename = os.path.join(saved_path,"dev.csv")
    train[["label","text"]].to_csv(train_filename,encoding="utf-8",sep="\t",index=False,header=None)
    dev[["label","text"]].to_csv(test_filename,encoding="utf-8",sep="\t",index=False,header=None)
    print("processing into formated files over") 
def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = 'glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        wordcount = 0
        for word, i in word2idx.items():
            embedding_matrix[i] = np.random.uniform(-0.25, +0.25, embed_dim)
            vec = word_vec.get(word)
            if vec is not None:
                wordcount = wordcount +1
                embedding_matrix[i] = vec
        print( "%.4f"% (wordcount/len(word2idx.items()) ))
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        # for fname in fnames:
        #     fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        #     lines = fin.readlines()
        #     fin.close()
        #     for i in range(0, len(lines), 3):
        #         text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        #         aspect = lines[i + 1].lower().strip()
        #         text_raw = text_left + " " + aspect + " " + text_right
        #         text += text_raw + " "
        # return text
        for fname in fnames:
            fins=pd.read_csv(fname,sep='\t',names=['label','text'])
            for i in range(len(fins['text'])):
                # text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                text +=fins['text'][i].lower()
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        # fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        # lines = fin.readlines()
        # fin.close()
        fin = open(fname+'.graph', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()
        all_data = []
        fins=pd.read_csv(fname,sep='\t',names=['label','text'])

        for i in range(len(fins['text'])):
            # text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            text=fins['text'][i].lower()            

            # aspect = lines[i + 1].lower().strip()
            
            polarity = fins['label'][i]
            text_indices = tokenizer.text_to_sequence(text)
            # context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            # aspect_indices = tokenizer.text_to_sequence(aspect)
            # left_indices = tokenizer.text_to_sequence(text_left)
            polarity = int(polarity)
            dependency_graph = idx2gragh[i]

            data = {
                'text_indices': text_indices,
                # 'context_indices': context_indices,
                # 'aspect_indices': aspect_indices,
                # 'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300):
        print('process dataset')
        # if dataset=='cr':
        #     print('process dataset')
        #     process(dataset)
        #     dependency_graph.process('./datasets/cr/train.csv')
        #     dependency_graph.process('./datasets/cr/dev.csv')
        # if dataset=='mr':
        #     process(dataset)
        #     dependency_graph.process('./datasets/mr/train.csv')
        #     dependency_graph.process('./datasets/mr/dev.csv')
        # if dataset=='mpqa':
        #     process(dataset)
        #     dependency_graph.process('./datasets/mpqa/train.csv')
        #     dependency_graph.process('./datasets/mpqa/dev.csv')
        # if dataset=='subj':
        #     process(dataset)
        #     dependency_graph.process('./datasets/subj/train.csv')
        #     dependency_graph.process('./datasets/subj/dev.csv')
        # if dataset=='sst2':
        #     dependency_graph.process('./datasets/sst2/train.csv')
        #     dependency_graph.process('./datasets/sst2/test.csv')
        # if dataset=='TREC':
        #     dependency_graph.process('./datasets/TREC/train.csv')
        #     dependency_graph.process('./datasets/TREC/test.csv')
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'cr': {
                'train': './datasets/cr/train.csv',
                'test': './datasets/cr/dev.csv'
            },
            'mr': {
                'train': './datasets/mr/train.csv',
                'test': './datasets/mr/dev.csv'
            },
            'mpqa': {
                'train': './datasets/mpqa/train.csv',
                'test': './datasets/mpqa/dev.csv'
            },
            'subj': {
                'train': './datasets/subj/train.csv',
                'test': './datasets/subj/dev.csv'
            },
            'sst2': {
                'train': './datasets/sst2/train.csv',
                'test': './datasets/sst2/test.csv'
            },
            'TREC': {
                'train': './datasets/TREC/train.csv',
                'test': './datasets/TREC/test.csv'
            },

        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        if os.path.exists(dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.embedding_matrix.shape
        print(fname[dataset]['train'])
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))
    