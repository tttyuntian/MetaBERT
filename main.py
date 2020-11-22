import argparse
import json
import gc

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
    parser.add_argument("--tasks", type=list, \
                        default=["mnli", "sst2", "qqp", "qnli"])
    parser.add_argument("--task_shared", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=1123)

    # I/O parameters
    parser.add_argument("--output_dir", type=str, default="./checkpoints", \
                        help="Output directory for model checkpoint.")

    # BERT parameters
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--padding", type=bool, default=True)
    parser.add_argument("--do_lower_case", type=bool, default=True)

    # MAML parameters
    parser.add_argument("--num_update_steps", type=int, default=5)
    parser.add_argument("--num_sample_tasks", type=int, default=8)
    parser.add_argument("--outer_learning_rate", type=float, default=5e-5)
    parser.add_argument("--inner_learning_rate", type=float, default=1e-3)

    # dataset preprocessing parameters
    parser.add_argument("--query_size", type=float, default=0.2, \
                        help="Proportion of query set in training set.")
    
    # training parameters
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--train_verbose", type=bool, default=False)
    
    args = parser.parse_args()

    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def get_train_steps(datasets, args):
    """ Get training steps for each task """
    return [dataset.num_rows // (args.train_batch_size*(args.num_update_steps+1)) for _, dataset in datasets.items()]

def get_classifiers(model, num_labels, args):
    return [Classifier(model, args.hidden_size, num_labels[task_id]) for task_id in range(len(args.tasks))]

def main(args):
    print("Loading datasets.")
    datasets      = {task:load_dataset("glue", args.task) for task in args.tasks}
    label_lists   = get_label_lists(datasets, args.tasks)
    num_labels    = get_num_labels(label_lists)

    print("Preprocessing datasets.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=args.do_lower_case)
    datasets  = preprocess(datasets, tokenizer, args)

    print("Retrieving training set.")
    train_datasets = get_split_datasets(datasets, "train", seed=args.seed)
    support_datasets, query_datasets = support_query_split(train_datasets, args.query_size)
    support_dataloaders = get_dataloaders(support_datasets, "support", args)
    query_dataloaders   = get_dataloaders(query_datasets, "query", args)

    print("Load BERT and Create classifiers.")
    model = BertModel.from_pretrained("bert-base-uncased")
    outer_optimizer = Adam(model.parameters(), lr=outer_learning_rate)
    classifiers = get_classifiers(model, num_labels, args)
    model.to(device)

    print("Start training!")
    train_steps_per_task = get_train_steps(support_datasets, args)
    for epoch_id in range(args.num_train_epochs):
        print("Start Epoch {}".format(epoch_id))
        model = maml(model, outer_optimizer, support_dataloaders, query_dataloaders, train_steps_per_task, device, args)

    print("Output checkpoint to /{}".format(args.output_dir))
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)