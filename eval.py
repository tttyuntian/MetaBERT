from transformers import BertConfig, BertModel, BertTokenizer
import argparse

from main import get_logger, get_classifiers
from datasets import load_dataset, load_metric
from modules.preprocess import get_label_lists, preprocess, get_num_labels, get_dataloaders
from transformers import AdamW


def parseArguments(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default="./metabert-small/")
    parser.add_argument("--tasks", nargs="+", \
                        default=["sst2"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--outer_learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_rows", type=int, default=-1, \
                        help="Number of datset rows loaded. -1 means whole dataset.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)

    return parser.parse_args()

def main(args):
    print("Loading Checkpoint from {}".format(args.checkpoint_path)) 
    model = BertModel.from_pretrained(args.checkpoint_path)     

    '''
    print("Loading Test Dataset")
    test = load_dataset("glue", args.eval_task, split="validation")
    print("Dataset Size: {}".format(len(test)))
    labels = get_label_lists({args.eval_task: test}, [args.eval_task])
    print(labels)
    print("Label Size: {}".format(len(labels)))
    num_labels = get_num_labels(labels)
    '''

    print("Preprocessing Fine Tune Data")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_datasets = {task:load_dataset("glue", task, split="train") for task in args.tasks}
    label_lists    = get_label_lists(train_datasets, args.tasks)
    num_labels     = get_num_labels(label_lists)
    train_datasets  = preprocess(train_datasets, tokenizer, args)


    print("Retrieving support and query sets.")
    support_datasets, query_datasets = support_query_split(train_datasets, 0.2)
    support_dataloaders = get_dataloaders(support_datasets, "support", args)
    query_dataloaders   = get_dataloaders(query_datasets, "query", args)

    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    classifiers = get_classifiers(num_labels, args)

    #for epoch in range(args.epochs):


    #fine tuning


    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    #test_datasets  = preprocess({args.task: test}, tokenizer, args)

    num_labels     = get_num_labels(labels)

    # fine tune the model 
    # train for preset epoch 

    
    print("hello")


if __name__ == "__main__":
    args = parseArguments()
    #logger = get_logger(args)
    main(args)