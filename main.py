import argparse
import json
import gc
import sys
import os
import logging

import numpy as np
import transformers
from transformers import BertModel, BertTokenizer
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset, load_metric

from modules.preprocess import get_label_lists, get_num_labels, preprocess
from modules.preprocess import get_split_datasets, support_query_split, get_dataloaders
from modules.meta_learning import maml
from modules.simple_classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def parse_arguments():
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--tasks", nargs="+", \
                        default=["mnli", "sst2", "qqp", "qnli"])
    parser.add_argument("--task_shared", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=1123)

    # I/O parameters
    parser.add_argument("--output_dir", type=str, default="../checkpoints", \
                        help="Output directory for model checkpoint.")
    parser.add_argument("--output_name", type=str, default="metabert")

    # BERT parameters
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--padding", type=bool, default=True)
    parser.add_argument("--do_lower_case", type=bool, default=True)

    # MAML parameters
    parser.add_argument("--num_update_steps", type=int, default=20)
    parser.add_argument("--num_sample_tasks", type=int, default=8)
    parser.add_argument("--outer_learning_rate", type=float, default=5e-5)
    parser.add_argument("--inner_learning_rate", type=float, default=1e-3)

    # dataset preprocessing parameters
    parser.add_argument("--query_size", type=float, default=0.2, \
                        help="Proportion of query set in training set.")
    parser.add_argument("--num_rows", type=int, default=-1, \
                        help="Number of datset rows loaded. -1 means whole dataset.")
    
    # training parameters
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--train_verbose", action="store_true")
    
    args = parser.parse_args()
    #print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def get_logger(args):
    os.makedirs("../logs/{}".format(args.output_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = '%(asctime)s %(levelname)s: %(message)s', \
            datefmt = '%m/%d %H:%M:%S %p', \
            filename = '../logs/{}/{}.log'.format(args.output_name, args.output_name), \
            filemode = 'w'
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logging.info("Argument {}: {}".format(arg, value))

def get_train_steps(dataloaders, args):
    """ Get training steps for each task """
    return [len(dataloader.dataset) // (args.train_batch_size*(args.num_update_steps+1)) for dataloader in dataloaders]

def get_classifiers(num_labels, args):
    return [Classifier(args.hidden_size, num_labels[task_id]) for task_id in range(len(args.tasks))]

def main(args):
    logger.info("Loading datasets.")
    train_datasets = {task:load_dataset("glue", task, split="train") for task in args.tasks}
    label_lists    = get_label_lists(train_datasets, args.tasks)
    num_labels     = get_num_labels(label_lists)

    logger.info("Preprocessing datasets.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=args.do_lower_case)
    train_datasets  = preprocess(train_datasets, tokenizer, args)

    logger.info("Retrieving support and query sets.")
    support_datasets, query_datasets = support_query_split(train_datasets, args.query_size)
    support_dataloaders = get_dataloaders(support_datasets, "support", args)
    query_dataloaders   = get_dataloaders(query_datasets, "query", args)
    
    logger.info("Load BERT and Create classifiers.")
    model = BertModel.from_pretrained("bert-base-uncased")
    outer_optimizer = Adam(model.parameters(), lr=args.outer_learning_rate)
    classifiers = get_classifiers(num_labels, args)
    #model.to(device)

    logger.info("Start training!")
    train_steps_per_task = get_train_steps(support_dataloaders, args)
    for epoch_id in range(args.num_train_epochs):
        logger.info("=" * 50)
        logger.info("Start Epoch {}".format(epoch_id))
        model, classifiers = maml(model, classifiers, outer_optimizer, support_dataloaders, query_dataloaders, train_steps_per_task, device, args)

    logger.info("Output checkpoint to /{}".format(args.output_dir))
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    model.save_pretrained(output_path)

if __name__ == "__main__":
    args = parse_arguments()
    logger = get_logger(args)
    logging_args(args)
    main(args)
