import re
import os
import abc
import torch
import pickle
import random
import numpy

from functools import reduce
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from torch.utils.data.dataloader import Dataset, DataLoader

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")


def text__cleaning(text):
    text = str(text).lower()
    text = re.sub(r"[!@#$%^&*()\-_+=\[{\]};:'\",<.>/?\\|~]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"\ss\s", " ", text)
    text = re.sub(r"\sre\s", " ", text)
    text = re.sub(r"\sve\s", " ", text)
    text = re.sub(r"\sd\s", " ", text)
    text = re.sub(r"\sll\s", " ", text)
    text = re.sub(r"\sm\s", " ", text)
    text = re.sub(r"\st\s", " ", text)
    
    return text.strip()


def load_data(args):
    data = {
        'train': [[], []],
        'validation': [[], []],
        'test': [[], []],
        "embeddings": numpy.zeros(0)
    }
    with open(args.data_dir + 'SICK.txt', "r", encoding="utf-8") as SICK_file:
        lines = SICK_file.readlines()
        for line in lines[1:]:
            elements = line.split("\t")
            sentence_A, sentence_B, relatedness_score = text__cleaning(elements[1]), text__cleaning(elements[2]), float(
                elements[4])
            SemEval_set = "validation" if elements[11].strip() == "TRIAL" else elements[11].strip().lower()
            
            data[SemEval_set][0].append((sentence_A, sentence_B))
            data[SemEval_set][1].append(relatedness_score / 5)
    return data


def sentence2vec(args):
    data = load_data(args)
    print("Loading word2vec from {}...".format(args.data_dir + 'GoogleNews-vectors-negative300.bin.gz'))
    word2vec = KeyedVectors.load_word2vec_format(args.data_dir + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
    print("Word2vec has been loaded.\n")
    
    stops = set(stopwords.words("english"))
    vocabulary = dict()
    inverse_vocabulary = ["<UNKNOWN>"]
    max_seq_len = 0
    
    for set_name in "train", "test", "validation":
        raw_X = data[set_name][0]
        Y = data[set_name][1]
        
        X = []
        for pair in raw_X:
            numerated_pair = []
            for sentence in pair:
                numerated_sentence = []
                word_list = sentence.split(" ")
                for word in word_list:
                    if word in stops and word not in word2vec:
                        continue
                    
                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        numerated_sentence.append(vocabulary[word])
                        inverse_vocabulary.append(word)
                    else:
                        numerated_sentence.append(vocabulary[word])
                numerated_pair.append(numerated_sentence)
                
                if len(numerated_sentence) > max_seq_len:
                    max_seq_len = len(numerated_sentence)
            
            X.append(numerated_pair)
        
        data[set_name] = [X, Y]
        data["max_seq_len"] = max_seq_len
    
    embeddings = 1 * numpy.random.randn(len(vocabulary) + 1, args.embedding_dim)
    for word, index in vocabulary.items():
        if word in word2vec:
            embeddings[index] = word2vec[word]
    embeddings[0] = 0
    data["embeddings"] = embeddings
    return data


class SICKDataset(Dataset):
    def __init__(self, data, phase):
        self.data = data[phase]
        self.max_len = data['max_seq_len']

    def pad_sentence(self, sentence):
        padded = []
        padding_len = self.max_len - len(sentence)
        padding = [0 for _ in range(padding_len)]
        padded.append(sentence + padding)
        return padded

    def __getitem__(self, index):
        sentence_pair = self.data[0][index]
        score = self.data[1][index]
        sentence_A_padded = self.pad_sentence(sentence_pair[0])
        sentence_B_padded = self.pad_sentence(sentence_pair[1])
        sample = {
            'sentence_A': torch.squeeze(torch.LongTensor(sentence_A_padded), dim=0),
            'sentence_B': torch.squeeze(torch.LongTensor(sentence_B_padded), dim=0),
            'score': torch.FloatTensor([score])
        }
        return sample

    def __len__(self):
        return len(self.data[0])


def load_iter(args):
    #data = sentence2vec(args)
    data = pickle.load(open(args.data_dir + 'data.pkl', 'rb'))
    train_dataset = SICKDataset(data, phase='train')
    val_dataset = SICKDataset(data, phase='validation')
    test_dataset = SICKDataset(data, phase='test')
    
    train_iter = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    val_iter = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
    test_iter = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    
    return data, train_iter, val_iter, test_iter
