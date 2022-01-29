import os
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from os.path import join
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List, Dict, Tuple


class SimCSEDataset(Dataset):
    def __init__(self, data, args, tokenizer, phase='train'):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.phase = phase
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        batch = self.data[index]
        if self.args.do_mlm and self.phase == 'train':
            mask_input_ids, mask_label = mask_tokens(self.args, self.tokenizer, batch['input_ids'])
            batch['mlm_input_ids'] = mask_input_ids
            batch['mlm_labels'] = mask_label
        
        return batch


def load_train_data_unsupervised(tokenizer, args):
    """
    获取无监督训练语料
    """
    logger.info('loading unsupervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-unsupervised.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    with open(args.train_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        # lines = lines[:100]  # debugging
        logger.info("len of train data:{}".format(len(lines)))
        for line in tqdm(lines):
            line = line.strip()
            feature = tokenizer([line, line], max_length=args.max_len, truncation=True, padding='max_length',
                                return_tensors='pt')
            feature_list.append(feature)
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_train_data_supervised(tokenizer, args):
    """
    获取NLI监督训练语料
    """
    logger.info('loading supervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-supervised.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    df = pd.read_csv(args.train_file, sep=',')
    logger.info("len of train data:{}".format(len(df)))
    rows = df.to_dict('reocrds')
    # rows = rows[:10000]
    for row in tqdm(rows):
        sent0 = row['sent0']
        sent1 = row['sent1']
        hard_neg = row['hard_neg']
        feature = tokenizer([sent0, sent1, hard_neg], max_length=args.max_len, truncation=True, padding='max_length',
                            return_tensors='pt')
        feature_list.append(feature)
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_eval_data(tokenizer, args, phase):
    """
    加载验证集或者测试集
    """
    assert phase in ['dev', 'test'], 'mode should in ["dev", "test"]'
    logger.info('loading {} data'.format(phase))
    output_path = os.path.dirname(args.output_path)
    eval_file_cache = join(output_path, '{}.pkl'.format(phase))
    if os.path.exists(eval_file_cache) and not args.overwrite_cache:
        with open(eval_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of {} data:{}".format(phase, len(feature_list)))
            return feature_list
    
    if phase == 'dev':
        eval_file = args.dev_file
    else:
        eval_file = args.test_file
    feature_list = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        logger.info("len of {} data:{}".format(phase, len(lines)))
        for line in tqdm(lines):
            line = line.strip().split("\t")
            score = float(line[4])
            data1 = tokenizer(line[5].strip(), max_length=args.max_len, truncation=True, padding='max_length',
                              return_tensors='pt')
            data2 = tokenizer(line[6].strip(), max_length=args.max_len, truncation=True, padding='max_length',
                              return_tensors='pt')
            
            feature_list.append((data1, data2, score))
    
    with open(eval_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_iter(args, tokenizer):
    assert args.train_mode in ['supervise', 'unsupervise'], \
        "train_mode should in ['supervise', 'unsupervise']"
    if args.train_mode == 'supervise':
        train_data = load_train_data_supervised(tokenizer, args)
    else:
        train_data = load_train_data_unsupervised(tokenizer, args)
    train_dataset = SimCSEDataset(train_data, args, tokenizer)
    dev_data = load_eval_data(tokenizer, args, 'dev')
    dev_dataset = SimCSEDataset(dev_data, args, tokenizer, phase='dev')
    
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                            num_workers=args.num_workers)
    dev_iter = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                          num_workers=args.num_workers)
    return train_iter, dev_iter


def load_predict_iter(args, tokenizer):
    test_data = load_eval_data(tokenizer, args, 'test')
    test_dataset = SimCSEDataset(test_data, args, tokenizer, phase='test')
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                           num_workers=args.num_workers)
    return test_iter


def mask_tokens(args, tokenizer, inputs, special_tokens_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    inputs = inputs.clone()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, args.mlm_prob)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()
    
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
