import os
import argparse

from data_loader import load_and_cache_examples
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_test:
        trainer.load_model()
        test_results = trainer.evaluate("test")
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(test_results.keys()):
                writer.write("{} = {:.4f}\n".format(key, test_results[key]))
        print(test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default='HPRD50', type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="./data/PPI/HPRD50", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="./output/PPI/HPRD50/No_entity_word/3_3e-5_20/", type=str, help="Path to model")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="Dev file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")
    parser.add_argument("--model_name_or_path", type=str, default="./pretrain_model/biobert-base-cased-v1.1",
                        help="Model Name or Path.")
    parser.add_argument("--seed", type=int, default=2024, help="random seed for initialization.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")

    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=256, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_character_len", default=20, type=int,
                        help="The maximum total input character length after tokenization.")

    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=20, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence.")

    parser.add_argument("--bilstm_layers", type=int, default=3, help="BiLSTM Layers")
    parser.add_argument("--n_heads", type=int, default=3, help="MultiHead Attention Num_Heads")

    parser.add_argument("--filter_num", type=int, default=3, help="filter number")
    parser.add_argument("--filter_sizes", type=str, default='3,4,5', help="filter sizes")

    parser.add_argument("--word_embedding_size", type=int, default=300, help="word embedding size")
    parser.add_argument("--word_vocab_size", type=int, default=83997, help="word vocab size")

    parser.add_argument("--char_dict_path", default="./data/char2id.json", type=str, help="The char dict path.")
    parser.add_argument("--char_embedding_size", type=int, default=30, help="char embedding size")
    parser.add_argument("--kernel_size", type=int, default='3,4,5', help="kernel size")
    parser.add_argument("--filter_number", type=int, default=3, help="filter number")
    parser.add_argument("--char_vocab_size", type=int, default=66, help="char vocab size")

    parser.add_argument("--pos_dict_path", default="./data/pos2id.json", type=str, help="The pos dict path.")
    parser.add_argument("--pos_embedding_size", type=int, default=40, help="pos embedding size")
    parser.add_argument("--pos_vocab_size", type=int, default=48, help="pos vocab size")

    parser.add_argument("--patience_num", type=int, default=10)
    parser.add_argument("--patience", type=float, default=0.00001)

    args = parser.parse_args()

    main(args)
