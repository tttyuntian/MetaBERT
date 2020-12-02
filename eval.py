from transformers import BertConfig, BertModel, BertTokenizer
import argparse

from main import get_logger, get_classifiers, get_train_steps, support_query_split
from datasets import load_dataset, load_metric
from modules.preprocess import get_label_lists, preprocess, get_num_labels, get_dataloaders
from transformers import AdamW
from modules.meta_learning import sample_task
from copy import deepcopy
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parseArguments(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default="./metabert-small/")
    parser.add_argument("--tasks", nargs="+", \
                        default=["sst2"])
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--outer_learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_rows", type=int, default=-1, \
                        help="Number of datset rows loaded. -1 means whole dataset.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--padding", type=bool, default=True)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--train_verbose", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_update_steps", type=int, default=20)
    parser.add_argument("--num_sample_tasks", type=int, default=8)
    parser.add_argument("--inner_learning_rate", type=float, default=1e-3)
    parser.add_argument("--config-path", type=str, default="./metebert-small-tuned/")

    return parser.parse_args()

def main(args):
    print("Loading Checkpoint from {}".format(args.checkpoint_path)) 
    model = BertModel.from_pretrained(args.checkpoint_path)     

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

    print("In Train Mode")

    model.train()
    classifiers = get_classifiers(num_labels, args)

    train_steps_per_task = get_train_steps(support_dataloaders, args)

    sample_task_ids = sample_task(train_steps_per_task, args)
    #print(sample_task_ids)
    #print(classifiers)
    classifier = classifiers[0]
    classifier.embedder = deepcopy(model)
    classifier.to(device)
    optimizer = AdamW(classifier.parameters(), lr=1e-5) #not sure if we should be optimizing the underlying BERT directly
    classifier.train()

    for epoch_id in range(args.num_train_epochs):
        print(epoch_id)
        #print(support_dataloaders)
        support_dataloader = support_dataloaders[0]

        all_loss = []
        print("starting to step")
        for step_id in range(args.num_update_steps):
            break
            batch = next(iter(support_dataloader))
            input_ids, attention_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
            print(labels)
            outputs = classifier(input_ids, attention_mask, token_type_ids, labels = labels)
            loss = outputs[1]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_loss.append(loss.item())

    ## UNCOMMENT THIS torch.save(classifier, args.config-path)

    print("Loading Test Dataset")
    test_datasets = {task:load_dataset("glue", task, split="validation") for task in args.tasks}
    #print("Dataset Size: {}".format(len(test_datasets[0])))
    test_label_lists    = get_label_lists(test_datasets, args.tasks)
    test_num_labels     = get_num_labels(test_label_lists)
    #print("Label Size: {}".format(len(test_label_lists)))

    #3) Set model.eval()
    #4) somehow use torch.no_grad() 
    #5) evaluate the results

    print("hello")


if __name__ == "__main__":
    args = parseArguments()
    #logger = get_logger(args)
    main(args)