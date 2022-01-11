import argparse


def args_init():
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help="gpu or cpu")
    parser.add_argument("--output_path", type=str, default='output')
    parser.add_argument("--lr", type=list, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--eval_step", type=int, default=50, help="every eval_step to evaluate model")
    parser.add_argument("--max_len", type=int, default=64, help="max length of input")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_lr_scheduler", type=bool, default=False)
    parser.add_argument("--lr_scheduler_type", default="linear", help="the scheduler type to use.",
                        choices=[
                            "linear",
                            "cosine",
                            "cosine_with_restarts",
                            "polynomial",
                            "constant",
                            "constant_with_warmup",
                        ], )
    parser.add_argument("--overwrite_cache", action='store_true', default=False, help="overwrite cache")
    
    # 文件位置
    parser.add_argument("--train_file", type=str, default="data/nli_for_simcse.csv")  # supervised
    # parser.add_argument("--train_file", type=str, default="data/wiki1m_for_simcse.txt")  # unsupervised
    parser.add_argument("--dev_file", type=str, default="data/stsbenchmark/sts-dev.csv")
    parser.add_argument("--test_file", type=str, default="data/stsbenchmark/sts-test.csv")
    parser.add_argument("--pretrain_model_path", type=str, default="bert-base-uncased")
    
    # 训练
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg", 'cls-mlp'],
                        default='cls-mlp',
                        help='pooler to use')  # 'cls' for unsupervised simcse and 'cls-mlp' for supervised
    parser.add_argument("--train_mode", type=str, default='supervise', choices=['unsupervise', 'supervise'],
                        help="unsupervise or supervise")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_predict", action='store_true', default=True)
    
    # MLM 参数
    parser.add_argument("--do_mlm", action='store_true', default=True)
    parser.add_argument("--mlm_weight", type=float, default=0.2)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    return parser
