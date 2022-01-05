import time
import torch
import datetime
import torch.utils.data
from torch import nn

import utils


def train(args, model, train_iter, val_iter, metric, optimizer, lr_scheduler, logger, scaler):
    print("Start training")
    start_time = time.time()
    losses = []
    for epoch in range(args.num_train_epochs):
        model.train()
        metric_logger = utils.MetricLogger(logger=logger, delimiter="  ")
        metric_logger.add_meter(
            "lr", utils.SmoothedValue(
                window_size=1, fmt="{value}"))
        metric_logger.add_meter(
            "sentence/s", utils.SmoothedValue(
                window_size=10, fmt="{value}"))
        
        header = "Epoch: [{}]".format(epoch + 1)
        i = 0
        for batch in metric_logger.log_every(train_iter, args.print_freq, header):
            start_time = time.time()
            batch.to(args.device)
            if 'roberta' in args.model_name_or_path:
                batch.pop('token_type_ids')
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss = model(**batch)[0]
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            lr_scheduler.step()
            batch_size = batch["input_ids"].shape[0]
            metric_logger.update(
                loss=loss.item(), lr=lr_scheduler.get_last_lr()[-1])
            metric_logger.meters["sentence/s"].update(batch_size /
                                                      (time.time() - start_time))
        
        pearson, spearmanr, combined_score = evaluate(args, model, val_iter, metric, logger)
    if args.output_dir:
        model.save_pretrained(args.output_dir)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    print("Training time {}".format(total_time_str))
    return pearson, spearmanr, combined_score


def evaluate(args, model, val_iter, metric, logger, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(logger=logger, delimiter="  ")
    header = "Test:"
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in metric_logger.log_every(val_iter, print_freq, header):
            batch.to(args.device)
            if 'roberta' in args.model_name_or_path:
                batch.pop('token_type_ids')
            
            labels = batch.pop("labels")
            logits = model(**batch)[0]
            loss = criterion(
                logits.reshape(-1, model.num_labels), labels.reshape(-1))
            metric_logger.update(loss=loss.item())
            metric.add_batch(predictions=logits, references=labels, )
    score = metric.compute()
    metric_logger.synchronize_between_processes()
    logger.info(" * Pearson {:.10f}".format(score['pearson']))
    logger.info(" * Spearmanr {:.10f}".format(score['spearmanr']))
    logger.info(" * Combined Score {:.10f}".format((score['pearson'] + score['spearmanr']) / 2))
    
    return score['pearson'], score['spearmanr'], (score['pearson'] + score['spearmanr']) / 2
