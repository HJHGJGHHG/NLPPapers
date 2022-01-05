import argparse


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Semantic Textual Similarity", add_help=add_help)
    parser.add_argument(
        "--data_cache_dir", default='./data', help="data cache dir.")
    parser.add_argument("--task_name", default="stsb",
                        help="the name of the glue task to train on.")
    parser.add_argument("--model_name_or_path", default="./roberta_base",
                        help="path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_length", type=int, default=128,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
    parser.add_argument("--num_train_epochs", default=4, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--workers", default=0, type=int,
                        help="number of data loading workers (default: 0)", )
    parser.add_argument("--lr", default=3e-5, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="weight decay (default: 1e-2)", dest="weight_decay", )
    parser.add_argument("--lr_scheduler_type", default="linear", help="the scheduler type to use.",
                        choices=[
                            "linear",
                            "cosine",
                            "cosine_with_restarts",
                            "polynomial",
                            "constant",
                            "constant_with_warmup",
                        ], )
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="number of steps for the warmup in the lr scheduler.", )
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output_dir", default=None, help="path where to save")
    parser.add_argument("--test_only", help="only test the model", action="store_true", )
    parser.add_argument("--seed", default=1234, type=int, help="a seed for reproducible training.")
    # Mixed precision training parameters
    parser.add_argument("--fp16", action="store_true", help="whether or not mixed precision training")
    
    return parser
