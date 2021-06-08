# -*- coding: utf-8 -*-
"""News_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rATTKS2rKIZlAoTof5ezEUEpMpX_j1Rt
"""

!pip install transformers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
 
import os

#DEFINITION OF CONSTANTS
FILE_PATH = '/content/drive/MyDrive/GPUs/Bert/DataSet/News_DataSet/11_CLASSES/MoreThan5k.csv'
RANDOM_SEED = 42
MAX_LEN = 200
BATCH_SIZE = 16
NCLASSES = 11
PRETRAINED_BERT_MODEL = 'bert-base-cased'
NHIDDENS = 768
PATIENCE = 3
NAME_CLASSES = []

#SETTING RANDOM VARIABLES
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#READING FILE
df = pd.read_csv(FILE_PATH)

#Other variables
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_MODEL)

class IMDBdataset(Dataset):
    
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = tokenizer.encode_plus(
            review,
            max_length = self.max_len,
            truncation = True,
            add_special_tokens = True,
            return_token_type_ids = False,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        } 

def data_loader(df, tokenizer, max_len, batch_size):
    dataset = IMDBdataset(
        reviews = df.headline.to_numpy(),
        labels = df.label.to_numpy(),
        tokenizer = tokenizer,
        max_len = MAX_LEN
    )

    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_validation, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

#Loader 
train_data_loader = data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
validation_data_loader = data_loader(df_validation, tokenizer, MAX_LEN, BATCH_SIZE)

class BERTArticleClassificator(nn.Module):

    def __init__(self, numClases):
        super(BERTArticleClassificator, self).__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_BERT_MODEL)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, NCLASSES)
    
    def forward(self, input_ids, attention_mask):
        outputs, cls_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = False
        )

        drop_out = self.drop(cls_output)
        output = self.linear(drop_out)

        return output

model = BERTArticleClassificator(NCLASSES)
model = model.to(device)

EPOCHS = 25 #Iteraciones de entrenamiento
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS #Batch_Size * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

#Iteración entrenamiento
def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(output, dim=1)
        loss = loss_fn(output, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double()/n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double()/n_examples, np.mean(losses)

test_acc_max, test_acc_index = 0, 0

for epoch in range(EPOCHS):
    print('Epoch {} de {}'.format(epoch+1, EPOCHS))
    print('-'*10)
    train_acc, train_loss = train_model(
          model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train)
    )
    
    test_acc, test_loss = eval_model(
          model, test_data_loader, loss_fn, device, len(df_test)
    )
    
    print('Entrenamiento: Loss: {}, accuracy: {}'.format(train_loss, train_acc))
    print('Validación: Loss: {}, accuracy: {}'.format(test_loss, test_acc))
    print('')

    if test_acc > test_acc_max:
      test_acc_max = test_acc
      test_acc_index = epoch
    
    elif epoch - test_acc_index >= PATIENCE:
      print('STOP AT {} EPOCH OBTAINED MAX {} FROM EPOCH {}'.format(epoch, train_acc_max, train_acc_index))
      break