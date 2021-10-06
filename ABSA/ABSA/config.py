import argparse

def args_initialization():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='atae_lstm', type=str)
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--epochs', default=15, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int)
    
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='device')
    parser.add_argument('--seed', default=2021, type=int, help='set seed for reproducibility')
    parser.add_argument('--val_dataset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')

    return parser.parse_args(args=[])