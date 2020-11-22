import gc

import numpy as np
import transformers
from transformers import BertModel, BertTokenizer
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSamplerß



def sample_task(train_steps_per_task, args):
    """ Get sample tasks based on Probability Proportional to Size (PPS) """
    sample_task_ids = []
    for task_id in range(len(args.tasks)):
        sample_task_ids += [task_id] * train_steps_per_task[task_id]
    sample_task_ids = np.random.choice(sampled_task_ids, len(sampled_task_ids), replace = False)
    return sample_task_ids

def maml(model, outer_optimizer, support_dataloaders, query_dataloaders, train_steps_per_task, device, args):
    model.train()
    
    sum_gradients = []
    sample_task_ids = sample_task(train_steps_per_task, args)
    
    for sample_task_id, task_id in enumerate(sample_task_ids):
        classifier = classifiers[task_id]
        classifier.embedder = deepcopy(model)
        classifier.to(device)
        inner_optimizer = Adam(classifier.parameters(), lr=args.inner_learning_rate)
        classifier.train()

        # Inner updates with support sets
        for step_id in range(num_update_steps):
            all_loss = []
            for inner_step, batch in enumerate(support_dataloaders[task_id]):
                input_ids, attention_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
                outputs = classifier(input_ids, attention_mask, token_type_ids, labels = labels)
                loss = outputs[1]
                loss.backward()
                inner_optimizer.step()
                inner_optimizer.zero_grad()
                all_loss.append(loss.item())
        if args.train_verbose:
            print("| inner_loss {:8.6f} |".format(np.mean(all_loss)))
        
        # Outer update with query set
        query_batch = iter(query_dataloader[task_id]).next()
        q_input_ids, q_attention_mask, q_token_type_ids, q_labels = tuple(t.to(device) for t in query_batch)
        q_outputs = classifier(q_input_ids, q_attention_mask, q_token_type_ids, labels=q_labels)
        
        # Compute the cumulative gradients of original BERT parameters
        q_loss = q_outputs[1]
        q_loss.backward()
        classifier.to(torch.device("cpu"))
        for i, (name, params) in enumerate(classifier.namsed_parameters()):
            if name.startswith("embedder"):
                if sample_task_id == 0:
                    sum_gradients.append(deepcopy(params.grad))
                else:
                    sum_gradients[i] += deepcopy(params.grad)
        
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
        
        del sum_gradients
        gc.collect()

    return model