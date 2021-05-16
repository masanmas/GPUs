import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

import os

#DEFINITION OF CONSTANTS
FILE_PATH = ''
RANDOM_SEED = 42
MAX_LEN = 200
BATCH_SIZE = 16
NCLASSES = 10
PRETRAINED_BERT_MODEL = 'bert-base-cased'
NHIDDENS = 768
NAME_CLASSES = []

#TEST FILE EXISTS
exit() if not os.path.exists(FILE_PATH) else print('LOAD DATA HERE') #COMPLETE#

#SETTING RANDOM VARIABLES
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = ('cuda:0' if torch.cuda.is_available() else 'CPU')
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

class REUTERSDataset()


