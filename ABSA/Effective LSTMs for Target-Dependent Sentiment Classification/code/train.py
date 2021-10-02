import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

from preprocessor import build_iter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train_preparations(args, model):
    # criterion
    criterion = nn.CrossEntropyLoss()
    # optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = args.optimizer(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.l2reg)
    return criterion, optimizer


def train(args, model, train_iter):
    criterion, optimizer = train_preparations(args, model)
    max_val_acc = 0
    max_val_f1 = 0
    global_step = 0
    path = None
    for epoch in range(args.epochs):
        n_correct, n_total, loss_total = 0, 0, 0
        # switch model to training mode
        model.train()
        for i_batch, sample_batched in enumerate(train_iter):
            global_step += 1
            # clear gradient accumulators
            
            inputs = [sample_batched[col].to(args.device) for col in args.inputs_cols]
            outputs = model(inputs)
            targets = sample_batched['polarity'].to(args.device)
            
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            loss_total += loss.item() * len(outputs)
            if global_step % args.log_step == 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info('steps: {0} loss: {1:.4f}, acc: {2:.4f}'.format(global_step, train_loss, train_acc))
        
        val_acc, val_f1 = evaluation(args, model, val_iter)
        logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            if not os.path.exists('state_dict'):
                os.mkdir('state_dict')
            path = 'state_dict/{0}_{1}_val_acc{2}'.format(args.model_name, args.dataset, round(val_acc, 4))
            torch.save(model.state_dict(), path)
            logger.info('>> saved: {}'.format(path))
        if val_f1 > max_val_f1:
            max_val_f1 = val_f1
    
    return path


def evaluation(args, model, val_iter):
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    # switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        for t_batch, t_sample_batched in enumerate(val_iter):
            t_inputs = [t_sample_batched[col].to(args.device) for col in args.inputs_cols]
            t_targets = t_sample_batched['polarity'].to(args.device)
            t_outputs = model(t_inputs)
            
            n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
            n_total += len(t_outputs)
            
            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
    
    acc = n_correct / n_total
    f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                          average='macro')
    return acc, f1


if __name__ == '__main__':
    # load iterator
    train_iter, test_iter, val_iter, args, embedding_matrix = build_iter()
    # model
    model = args.model_class(embedding_matrix, args).to(args.device)
    # start training
    best_model_path = train(args, model, train_iter)
    
    # test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_acc, test_f1 = evaluation(args, model, test_iter)
    logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
