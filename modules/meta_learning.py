import gc
import sys
from copy import deepcopy
import logging

import numpy as np
import transformers
from transformers import BertModel, BertTokenizer
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


def sample_task(train_steps_per_task, args):
    """ Get sample tasks based on Probability Proportional to Size (PPS) """
    sample_task_ids = []
    for task_id in range(len(args.tasks)):
        sample_task_ids += [task_id] * train_steps_per_task[task_id]
    sample_task_ids = np.random.choice(sample_task_ids, len(sample_task_ids), replace = False)
    return sample_task_ids

def maml(model, classifiers, outer_optimizer, support_dataloaders, query_dataloaders, train_steps_per_task, device, args):
    model.train()
    
    sum_gradients = []
    sample_task_ids = sample_task(train_steps_per_task, args)
    
    for sample_task_id, task_id in enumerate(sample_task_ids):
        logger.info("Sample tasks: {}/{}".format(sample_task_id, len(sample_task_ids)))
        classifier = classifiers[task_id]
        classifier.embedder = deepcopy(model)
        classifier.to(device)
        inner_optimizer = Adam(classifier.parameters(), lr=args.inner_learning_rate)
        classifier.train()
        
        all_loss = []
        support_dataloader = support_dataloaders[task_id]
        for step_id in range(args.num_update_steps):
            for _ in range(args.grad_acc_step):
                # Deal with small batch size with gradient accumulation
                batch = next(iter(support_dataloader))
                input_ids, attention_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
                outputs = classifier(input_ids, attention_mask, token_type_ids, labels = labels)
                loss = outputs[1]
                loss = loss / args.grad_acc_step    # # Scale the loss to the mean of the accumulated batch size
                loss.backward()
                all_loss.append(loss.item())

            inner_optimizer.step()
            inner_optimizer.zero_grad()
            
        if args.train_verbose and sample_task_id % args.report_step == (args.report_step-1):
            logger.info("| sample_task_id {:10d} | inner_loss {:8.6f} |".format(sample_task_id, np.mean(all_loss)))
        
        # Outer update with query set
        for _ in range(args.grad_acc_step):
            classifier.to(device)
            query_batch = next(iter(query_dataloaders[task_id]))
            q_input_ids, q_attention_mask, q_token_type_ids, q_labels = tuple(t.to(device) for t in query_batch)
            q_outputs = classifier(q_input_ids, q_attention_mask, q_token_type_ids, labels=q_labels)
            
            # Compute the cumulative gradients of original BERT parameters
            q_loss = q_outputs[1]
            q_loss = q_loss / args.grad_acc_step    # Scale the loss to the mean of the accumulated batch size
            q_loss.backward()
            classifier.to(torch.device("cpu"))
            gradient_index = 0
            for i, (name, params) in enumerate(classifier.named_parameters()):
                if name.startswith("embedder"):
                    if sample_task_id % args.num_sample_tasks == 0:
                        sum_gradients.append(deepcopy(params.grad))
                    else:
                        sum_gradients[gradient_index] += deepcopy(params.grad)
                        gradient_index += 1
            
        # Update BERT parameters after sampling num_sample_tasks
        if sample_task_id % args.num_sample_tasks == (args.num_sample_tasks-1):
            # Compute average gradient across tasks
            for i in range(len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / args.num_sample_tasks
            
            # Assign gradients for original BERT model and Update weights
            for i, params in enumerate(model.parameters()):
                params.grad = sum_gradients[i]
            
            outer_optimizer.step()
            outer_optimizer.zero_grad()
            sum_gradients = []
        
        # Update this classifier
        classifier.embedder = None
        #classifiers[task_id] = deepcopy(classifier)

        gc.collect()
        torch.cuda.empty_cache()

    return model, classifiers
