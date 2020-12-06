import argparse
import json
import sys
import os
import logging
from copy import deepcopy

import numpy as np
import transformers
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import AdamW
import torch
from datasets import load_dataset, load_metric
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

from modules.preprocess import get_label_lists, preprocess, get_num_labels, get_dataloaders
from modules.simple_classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--task", type=str, required=True, \
                        help="Evaluation GLUE task.")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--text_embedder", type=str, default="bert-base-uncased")

    # I/O parameters
    parser.add_argument("--load_path", type=str, required=True, \
                        help="Directory of the pretrained checkpoint")
    parser.add_argument("--ckpt_output_dir", type=str, required=True, \
                        help="Output directory for finetuned checkpoint")
    parser.add_argument("--ckpt_output_name", type=str, required=True, \
                        help="Output name for finetuned checkpoint")
    
    # MetaBERT parameters
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--padding", type=bool, default=True)
    parser.add_argument("--do_lower_case", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Finetuning parameters
    parser.add_argument("--num_rows", type=int, default=-1, \
                        help="Number of datset rows loaded. -1 means whole dataset.")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--train_verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=500)

    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("../logs/{}".format(args.ckpt_output_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = '%(asctime)s %(levelname)s: %(message)s', \
            datefmt = '%m/%d %H:%M:%S %p', \
            filename = '../logs/{}/{}.log'.format(args.ckpt_output_name, args.ckpt_output_name), \
            filemode = 'w'
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def finetune(classifier, train_dataloader, optimizer, epoch_id, args):
    losses = []
    classifier.train()
    
    for iter_id, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids, attention_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
        _, loss = classifier(input_ids, attention_mask, token_type_ids, labels = labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
        if args.train_verbose and iter_id % args.report_step == 0:
            logger.info(f"| Epoch {epoch_id:6d} | iter {iter_id:8d} | train_loss {float(np.mean(losses)):8.5f} |")
            losses = []

def evaluate(classifier, eval_dataloader, epoch_id, task):
    classifier.eval()
    eval_metric_1  = None   # some tasks have two metrics
    eval_metric_2  = None
    eval_accuracy  = 0      
    eval_data_size = 0      
    pred_list = []          
    true_list = []

    for iter_id, batch in enumerate(eval_dataloader):
        input_ids, attention_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            logits, _ = classifier(input_ids, attention_mask, token_type_ids, labels = labels)
        
        logits = logits.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        iter_eval_accuracy = np.sum(np.argmax(logits, axis=1) == labels)
        eval_accuracy  += iter_eval_accuracy
        eval_data_size += input_ids.size(0)

        if task == "stsb":
            pred_list.append(logits)
        elif task in ["qqp", "mrpc"]:
            pred = np.argmax(logits, axis=1)
            pred_list.append(np.minimum(pred, np.ones_like(pred)))
        else:
            pred_list.append(np.argmax(logits, axis=1))
        true_list.append(labels)

    if task == "cola":
        pred = np.concatenate(pred_list)
        true = np.concatenate(true_list)
        eval_metric_1 = matthews_corrcoef(pred, true)
    elif task == "stsb":
        pred = np.squeeze(np.concatenate(pred_list))
        true = np.squeeze(np.concatenate(true_list))
        eval_metric_1 = pearsonr(pred, true)[0]
        eval_metric_2 = spearmanr(pred, true)[0]
    else:
        eval_metric_1 = eval_accuracy / eval_data_size

    if task in ["qqp", "mrpc"]:
        pred = np.concatenate(pred_list)
        true = np.concatenate(true_list)
        eval_metric_2 = f1_score(true, pred, average="binary")

    if task == "cola":
        logger.info(f"| Epoch {epoch_id:6d} | eval_mcc {eval_metric_1:6.4f} |")
    elif task in ["qqp", "mrpc"]:
        logger.info(f"| Epoch {epoch_id:6d} | eval_acc {eval_metric_1:6.4f} | eval_f1 {eval_metric_2:6.4f} |")
    elif task == "stsb":
        logger.info(f"| Epoch {epoch_id:6d} | eval_pearson {eval_metric_1:6.4f} | eval_spearman {eval_metric_2:6.4f} |")
    else:
        logger.info(f"| Epoch {epoch_id:6d} | eval_acc {eval_metric_1:6.4f} |")

    return eval_metric_1, eval_metric_2

def main(args):
    logger.info("Preprocessing training and evaluatation data")
    tokenizer = BertTokenizer.from_pretrained(args.text_embedder, do_lower_case=args.do_lower_case)
    
    train_datasets = {task:load_dataset("glue", task, split="train") for task in [args.task]}
    label_lists    = get_label_lists(train_datasets, [args.task])
    num_labels     = get_num_labels(label_lists)
    train_datasets = preprocess(train_datasets, tokenizer, args)
    print(train_datasets)
    train_dataloaders = get_dataloaders(train_datasets, "train", args, is_eval=True)
    train_dataloader  = train_dataloaders[0]
    print("hello")
    print(len(train_dataloader))

    eval_datasets  = {task:load_dataset("glue", task, split="validation") for task in [args.task]}
    eval_datasets  = preprocess(eval_datasets, tokenizer, args)
    eval_dataloaders  = get_dataloaders(eval_datasets, "validation", args, is_eval=True)
    eval_dataloader   = eval_dataloaders[0]

    logger.info("Retrieve classifier with pretrained checkpoint from {}.".format(args.load_path))
    classifier = Classifier(args.hidden_size, num_labels[0], args.dropout)
    if args.load:
        classifier.embedder = BertModel.from_pretrained(args.load_path)
    else:
        classifier.embedder = BertModel.from_pretrained(args.text_embedder)
    classifier.to(device)
    optimizer = AdamW(classifier.parameters(), lr=args.learning_rate)
    
    logger.info("Start finetuning.")
    task = args.task
    best_metric_1 = -1
    best_metric_2 = -1
    
    for epoch_id in range(args.num_epochs):
        logger.info("="*70)
        # Finetune
        finetune(classifier, train_dataloader, optimizer, epoch_id, args)
        
        # Evaluate
        metric_1, metric_2 = evaluate(classifier, eval_dataloader, epoch_id, task)

        # Early stopping
        if metric_1 > best_metric_1:
            logger.info("*"*70)
            
            logger.info(f"Best model found for {args.task}.")
            best_metric_1 = metric_1
            if task in ["qqp", "mrpc"]:
                best_metric_2 = metric_2
            elif task == "stsb":
                best_metric_2 = metric_2
            
            logger.info("Output checkpoint to /{}".format(args.ckpt_output_dir))
            os.makedirs(args.ckpt_output_dir, exist_ok=True)
            output_path = os.path.join(args.ckpt_output_dir, args.ckpt_output_name)
            torch.save(classifier.state_dict(), "{}.pt".format(output_path))
            
            logger.info("*"*70)

        logger.info("="*70)

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)
    logger.info("Device: {}".format(device))
    
    main(args)
