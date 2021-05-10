#V1.0 This version is made with tensor flow library

import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split

from textwrap import wrap

#DEFINICION DE CONSTANTES
CKPT_PATH = "./Bert_Model.ckpt"
DATASET = "../DataSet/News_DataSet/3_CLASSES/News6k.csv"
RANDOM_SEED = 42
MAX_LEN = 200
BATCH_SIZE = 16
EPOCHS = 10

PRETRAINED_BERT_MODEL = 'bert-base-cased'
BERT_HIDDENS = 768
NCLASES = 2

np.random.seed(RANDOM_SEED)

try:
	df = pd.read_csv(DATASET)
except:
	print("ERROR: 001 Failed reading DataSet From", DATASET)
	exit()

try:
	df_labels = df['label1']
	df_data = df['label2']
except:
	print("ERROR: 002 Failed reading labels and data, name label1 to label column and label2 to data column")
	exit()


tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_MODEL)
bert = BertModel.from_pretrained(PRETRAINED_BERT_MODEL)

#SAMPLE DEL PROCESO DE TOKENIZACIÓN
sample_txt = "Places to visit with your son"
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(
	'Frase: ', sample_txt,
	'\nTokens: ', tokens,
	'\nToken_IDS: ', token_ids
)

encoder = tokenizer.encode_plus(
    sample_txt,
    max_length = 15,
    truncation = True,
    add_special_tokens = True,
    return_token_type_ids = False,
    padding = 'max_length',
    return_attention_mask = True,
    return_tensors = 'pt' 
)

print(
	encoder.keys(),
	encoder['input_ids']
)

#FUNCION PARA CREAR EL MODELO TENSOR FLOW DOS CAPAS, HIDDEN CON TAMAÑO BERT_NHIDDENS y CAPA DE SALIDA TAMAÑO NCLASES
def create_model():
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(1,BERT_HIDDENS)),
		keras.layers.Dense(NCLASES, activation='softmax')
	])

	model.compile(
		optimizer='adam',
		loss='parse_categorical_crossentropy',
		metrics=['accuracy'],
	)

	return model



outputs, cls_output = bert(encoder['input_ids'], encoder['attention_mask'], return_dict=False)

print(cls_output.shape)

model = create_model()
print(model.summary())

model.fit(cls_output, [1])
