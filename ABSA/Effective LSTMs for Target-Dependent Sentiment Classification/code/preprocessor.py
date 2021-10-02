import os
import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import config
from models import LSTM, TD_LSTM, TC_LSTM


def add_arguments(args):
    """
    根据参数解析器添加对应参数
    """
    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        #        'atae_lstm': ATAE_LSTM,
    }
    dataset_files = {
        'twitter': {
            'train': 'data/acl-14-short-data/train.raw',
            'test': 'data/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': 'data/semeval14/Restaurants_Train.xml.seg',
            'test': 'data/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': 'data/semeval14/Laptops_Train.xml.seg',
            'test': 'data/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tc_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    args.model_class = model_classes[args.model_name]
    args.dataset_file = dataset_files[args.dataset]
    args.inputs_cols = input_colses[args.model_name]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    
    return args


def load_word_vec(path, word2idx=None):
    """
    加载预训练的词向量
    """
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    """
    填充与截断，可以选择在头部还是尾部操作
    """
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


def build_tokenizer(fnames, max_seq_len, dat_fname):
    """
    构建分词器
    """
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            f = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = f.readlines()
            f.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    """
    基于词典精简预训练词向量，构造本任务的词向量矩阵
    """
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './model/glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else 'D:/python/pyprojects/NLP/model/glove.42B.300d/glove.42B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1
    
    def fit_on_text(self, text):
        """
        构建词典
        """
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
    
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        """
        将句子转换为字典中的序列，参数 reverse 表示是否反转，LSTM_R 的输入序列与自然序列是相反的
        """
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        f = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = f.readlines()
        f.close()
        
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            
            # 带目标词句子的索引序列 以及 不带目标词的索引序列
            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            # 左句子（第一个字到最后一个目标词）
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            # 右句子（最后一个字到第一个目标词）
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            # 目标词的索引
            aspect_indices = tokenizer.text_to_sequence(aspect)
            # 左句子长度
            left_context_len = np.sum(text_left_indices != 0)
            # 目标词长度 以及 目标词的位置
            aspect_len = np.sum(aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            
            polarity = int(polarity) + 1
            
            data = {
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
            }
            
            all_data.append(data)
        self.data = all_data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


def build_iter():
    args = config.args_initialization()
    args = add_arguments(args)
    set_seed(args)
    args.tokenizer = build_tokenizer(
        fnames=[args.dataset_file['train'], args.dataset_file['test']],
        max_seq_len=args.max_seq_len,
        dat_fname='{0}_tokenizer.dat'.format(args.dataset))
    embedding_matrix = build_embedding_matrix(
        word2idx=args.tokenizer.word2idx,
        embed_dim=args.embed_dim,
        dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(args.embed_dim), args.dataset))
    
    # build dataset
    train_dataset = ABSADataset(args.dataset_file['train'], args.tokenizer)
    test_dataset = ABSADataset(args.dataset_file['test'], args.tokenizer)
    # split train and val
    assert 0 <= args.val_dataset_ratio < 1
    if args.val_dataset_ratio > 0:
        val_dataset_len = int(len(train_dataset) * args.val_dataset_ratio)
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - val_dataset_len, val_dataset_len))
    else:
        val_dataset = test_dataset
    
    train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_iter = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    val_iter = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    
    return train_iter, test_iter, val_iter, args, embedding_matrix


def set_seed(args):
    # 设定种子保证可复现性
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
