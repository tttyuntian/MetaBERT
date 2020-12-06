import numpy as np
import transformers
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset

task_to_keys = { \
    "cola": ("sentence", None),         \
    "mnli": ("premise", "hypothesis"),  \
    "mrpc": ("sentence1", "sentence2"), \
    "qnli": ("question", "sentence"),   \
    "qqp" : ("question1", "question2"), \
    "rte" : ("sentence1", "sentence2"), \
    "sst2": ("sentence", None),         \
    "stsb": ("sentence1", "sentence2"), \
    "wnli": ("sentence1", "sentence2")  \
}

"""
task_cluster_dict = { \
    "mrpc": 0, \
    "cola": 1, \
    "mnli": 0, \
    "sst2": 1, \
    "rte" : 0, \
    "wnli": 0, \
    "qqp" : 0, \
    "qnli": 2, \
    "stsb": 3  \
}
task_clusters = [task_cluster_dict[task] for task in args.tasks] if args.task_shared else None
"""

def get_label_lists(datasets, tasks):
    """ Get a list of tasks' labels' names """
    label_lists = []
    for task in tasks:
        is_regression = task == "stsb"
        if is_regression:
            label_lists.append([None])
        else:
            label_lists.append(datasets[task].features["label"].names)
    return label_lists

def get_num_labels(label_lists):
    """ Get a list of number of labels for the tasks """
    return [len(label_list) for label_list in label_lists]

def preprocess(datasets, tokenizer, args):
    """ Preprocess every dataset """
    def preprocess_function(examples):
        """ Helper function to tokenize() the tokens """
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*inputs, padding=args.padding, max_length=args.max_length, truncation=True)
        return result

    for task, dataset in datasets.items():
        sentence1_key, sentence2_key = task_to_keys[task]
        datasets[task] = datasets[task].map(preprocess_function, batched=True)
    return datasets

def get_split_datasets(datasets, split="train", seed=None):
    """ Get datasets of specific split """
    if split == "train":
        split_datasets = {task:dataset[split].shuffle(seed=seed) for task, dataset in datasets.items()}
    else:
        split_datasets = {task:dataset[split] for task, dataset in datasets.items()}
    return split_datasets

def support_query_split(datasets, query_size):
    """ Carry out support set and query set split from the training datasets """
    support_datasets = {}
    query_datasets   = {}
    for task, dataset in datasets.items():
        support_query_split    = dataset.train_test_split(test_size=query_size)
        support_datasets[task] = support_query_split["train"]
        query_datasets[task]   = support_query_split["test"]
    return support_datasets, query_datasets

def get_dataloaders(datasets, split, args, is_eval=False):
    """ Convert datasets into torch.utils.data.DataLoader """
    dataloaders = []
    for task, dataset in datasets.items():
        num_rows = dataset.num_rows if args.num_rows == -1 else args.num_rows
        all_input_ids      = np.zeros([num_rows, args.max_length])
        all_attention_mask = np.zeros([num_rows, args.max_length])
        all_token_type_ids = np.zeros([num_rows, args.max_length])
        for i in range(num_rows):
            features = dataset[i]
            curr_len = len(features["attention_mask"])
            all_input_ids[i,:curr_len]      = features["input_ids"]
            all_attention_mask[i,:curr_len] = features["attention_mask"]
            all_token_type_ids[i,:curr_len] = features["token_type_ids"]
        all_input_ids      = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        all_label          = torch.tensor(dataset[:num_rows]["label"], dtype=torch.long)
        if is_eval and args.task == "stsb":
            all_label = all_label.float()
        
        data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
        if split in ["train", "support"]:
            sampler    = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        else:
            sampler    = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)
        dataloaders.append(dataloader)
    return dataloaders
