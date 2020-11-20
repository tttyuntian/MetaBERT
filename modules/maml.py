""" MAML with BERT """

from copy import deepcopy
import gc

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metric import accuracy_score



class MAML_BERT(nn.Module):
    
    def __init__(self, args):
        super(MAML_BERT, self).__init__()

        self.num_labels = args.num_labels
        self.inner_batch_size  args.inner_batch_size
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.outer_batch_size = args.outer_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.bert_model = args.bert_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BertForSequenceClassification
        

