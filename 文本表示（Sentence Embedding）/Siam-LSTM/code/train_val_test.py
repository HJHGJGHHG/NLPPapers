import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adadelta, Adam
from transformers import get_linear_schedule_with_warmup


def train(args, model, train_iter, val_iter, metric):
    # optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #optimizer = Adadelta(optimizer_grouped_parameters, lr=args.lr)
    optimizer = Adam(optimizer_grouped_parameters, lr=args.lr)
    
    # scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_iter) * args.epochs,
        num_training_steps=len(train_iter) * args.epochs
    )
    
    for epoch in range(args.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
        model = model.train()
        losses = []
        for step, batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()
            sentence_A = batch['sentence_A'].to(args.device)
            sentence_B = batch['sentence_B'].to(args.device)
            score = torch.squeeze(batch['score']).to(args.device)
            
            loss = model(sentence_A, sentence_B, score)[0]
            losses.append(loss.item())
            loss.backward()
            
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if (step + 1) % 10 == 0:
                print("Epoch: {}  {}/{}    MSE Loss: {}".format(epoch + 1, step + 1, len(train_iter), loss.item()))
        # 一个Epoch训练完毕，输出train_loss
        print('Epoch: {0}   Average Train MSE Loss: {1:>5.6}'.format(epoch + 1, np.mean(losses)))
        eval_model(args, model, val_iter, metric)
    # 训练结束


def eval_model(args, model, val_iter, metric):
    with torch.no_grad():
        model.eval()
        test_loss = []
        for batch in val_iter:
            sentence_A = batch['sentence_A'].to(args.device)
            sentence_B = batch['sentence_B'].to(args.device)
            score = torch.squeeze(batch['score']).to(args.device)
            outputs = model(sentence_A, sentence_B, score)
            test_loss.append(outputs[0].item())
            print(outputs[1])
            print(score)
            metric.add_batch(predictions=outputs[1], references=score, )
        result = metric.compute()
        print('Average Val MSE Loss: {0:>5.6}'.format(np.mean(test_loss)))
        print(" * Pearson {:.10f}".format(result['pearson']))
        print(" * Spearmanr {:.10f}".format(result['spearmanr']))
        
        
