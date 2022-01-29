import os
import torch
import pickle
from tqdm import tqdm
from os.path import join
from loguru import logger
from torch.utils.data import Dataset, DataLoader


def prepare_template(args, tokenizer):
    # template 1
    template1 = args.template1
    assert ' ' not in template1
    template1 = template1.replace('*mask*', tokenizer.mask_token) \
        .replace('*sep+*', '') \
        .replace('*model*', '').replace('*sent_0*', ' ')
    template1 = template1.split(' ')
    args.bs = template1[0].replace('_', ' ')
    args.es = template1[1].replace('_', ' ')
    
    # template 2
    template2 = args.template2
    assert ' ' not in template2
    template2 = template2.replace('*mask*', tokenizer.mask_token) \
        .replace('*sep+*', '') \
        .replace('*model*', '').replace('*sent_0*', ' ')
    template2 = template2.split(' ')
    args.bs2 = template2[0].replace('_', ' ')
    args.es2 = template2[1].replace('_', ' ')
    return args


def pad(ids, max_len):
    assert len(ids) <= max_len
    ids += [0] * (max_len - len(ids))
    return ids


def load_train_data_unsupervised(args, tokenizer):
    """
    获取无监督训练语料
    """
    logger.info('loading unsupervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'unsupervised-train-dataset.pkl')
    args = prepare_template(args, tokenizer)
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            train_dataset = pickle.load(f)
            return args, train_dataset
    
    bs = tokenizer.encode(args.bs)[:-1]
    es = tokenizer.encode(args.es)[1:]  # remove model or bos
    
    bs2 = tokenizer.encode(args.bs2)[:-1]
    es2 = tokenizer.encode(args.es2)[1:]  # remove model or bos
    feature_list = {'input_ids': [], 'attention_mask': []}
    with open(args.train_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        # lines = lines[:100]  # debugging
        logger.info("len of train data:{}".format(len(lines)))
        for line in tqdm(lines):
            text = line.strip()
            if text is None:
                text = ""
            original_input_ids = tokenizer.encode(text, add_special_tokens=False)[:args.max_len]
            ml = args.max_len + max(len(bs) + len(es), len(bs2) + len(es2))
            input_ids1 = bs + original_input_ids + es
            input_ids2 = bs2 + original_input_ids + es2
            attention_mask1 = len(input_ids1) * [1] + (ml - len(input_ids1)) * [0]
            attention_mask2 = len(input_ids2) * [1] + (ml - len(input_ids2)) * [0]
            feature_list['input_ids'].append([pad(input_ids1, ml), pad(input_ids2, ml)])
            feature_list['attention_mask'].append([attention_mask1, attention_mask2])
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return args, feature_list


def load_iter(args, tokenizer, model):
    args, train_data = load_train_data_unsupervised(args, tokenizer)
    train_dataset = PromptBertDataset(train_data, args, phase='train')
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    model.mask_token_id = tokenizer.mask_token_id
    model.pad_token_id = tokenizer.pad_token_id
    model.bos = tokenizer.encode('')[0]
    model.eos = tokenizer.encode('')[1]
    model.bs = tokenizer.encode(args.bs, add_special_tokens=False)
    model.es = tokenizer.encode(args.es, add_special_tokens=False)
    
    model.template1 = tokenizer.encode(args.bs + args.es)
    model.template2 = tokenizer.encode(args.bs2 + args.es2)
    model.to(args.device)
    
    dev_data = load_eval_data(tokenizer, args, 'dev')
    dev_dataset = PromptBertDataset(dev_data, args, phase='dev')
    dev_iter = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    return args, train_iter, dev_iter, model


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
    bs = tokenizer.encode(args.bs)[:-1]
    es = tokenizer.encode(args.es)[1:]  # remove model or bos
    feature_list = {'input_ids': [], 'attention_mask': [], 'score': []}
    with open(eval_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        logger.info("len of {} data:{}".format(phase, len(lines)))
        for line in tqdm(lines):
            line = line.strip().split("\t")
            score = float(line[4])
            original_input_ids1 = tokenizer.encode(line[5].strip(), add_special_tokens=False)[:args.max_len]
            original_input_ids2 = tokenizer.encode(line[6].strip(), add_special_tokens=False)[:args.max_len]
            ml = args.max_len + len(bs) + len(es)
            input_ids1 = bs + original_input_ids1 + es
            input_ids2 = bs + original_input_ids2 + es
            attention_mask1 = len(input_ids1) * [1] + (ml - len(input_ids1)) * [0]
            attention_mask2 = len(input_ids2) * [1] + (ml - len(input_ids2)) * [0]
            feature_list['input_ids'].append([pad(input_ids1, ml), pad(input_ids2, ml)])
            feature_list['attention_mask'].append([attention_mask1, attention_mask2])
            feature_list['score'].append(score)
    with open(eval_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


class PromptBertDataset(Dataset):
    def __init__(self, data, args, phase='train'):
        self.data = data
        self.args = args
        self.phase = phase
    
    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, index):
        input_ids = torch.LongTensor(self.data['input_ids'][index])
        attention_mask = torch.LongTensor(self.data['attention_mask'][index])
        data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if self.phase != 'train':
            data['score'] = torch.FloatTensor([self.data['score'][index]])
        return data
