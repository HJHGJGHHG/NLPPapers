import os
import time
import errno
import torch
import random
import datetime
import numpy as np
import torch.utils.data
from torch import nn
from collections import defaultdict, deque
from datasets import load_dataset, load_metric
from transformers import AdamW, DataCollatorWithPadding, get_scheduler
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta import RobertaTokenizer, RobertaForSequenceClassification

from models import RoBerta_CLS

task_to_keys = {
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        return
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value, )


class MetricLogger(object):
    def __init__(self, logger, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}",
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB, ))
                    self.logger.info(log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB, ))
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time), ))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {}".format(header, total_time_str))


def load_data(args, tokenizer):
    print("Loading data")
    raw_datasets = load_dataset(
        "dataset.py", args.task_name, cache_dir=args.data_cache_dir)
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    
    def preprocess_function(examples):
        texts = ((examples[sentence1_key],) if sentence2_key is None else
                 (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(
            *texts, padding=False, max_length=args.max_length, truncation=True)
        
        if "label" in examples:
            result["labels"] = examples["label"]
        return result
    
    train_ds = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on train dataset",
        new_fingerprint=f"train_tokenized_dataset_{args.task_name}", )
    validation_ds = raw_datasets["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Running tokenizer on validation dataset",
        new_fingerprint=f"validation_tokenized_dataset_{args.task_name}", )
    train_sampler = torch.utils.data.SequentialSampler(train_ds)
    validation_sampler = torch.utils.data.SequentialSampler(validation_ds)
    
    return train_ds, validation_ds, train_sampler, validation_sampler


def load_iter(args, tokenizer):
    collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if args.fp16 else None))
    train_dataset, validation_dataset, train_sampler, validation_sampler = load_data(args, tokenizer)
    train_iter = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=collate_fn, )
    
    validation_iter = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        sampler=validation_sampler,
        num_workers=args.workers,
        collate_fn=collate_fn, )
    
    return train_iter, validation_iter


def preparations(args):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    train_iter, validation_iter = load_iter(args, tokenizer)
    print("Creating model")
    model = RoBerta_CLS.from_pretrained(args.model_name_or_path, num_labels=1)  # regression
    device = torch.device(args.device)
    model.to(device)

    print("Creating criterion")
    criterion = nn.MSELoss()

    print("Creating optimizer")
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         },
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      betas=(0.9, 0.999),
                      eps=1e-8
                      )

    print("Creating lr_scheduler")
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_iter), )

    metric = load_metric("metric.py", "stsb")
    
    return train_iter, validation_iter, model, criterion, optimizer, lr_scheduler, metric