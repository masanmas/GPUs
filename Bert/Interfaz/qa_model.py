import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import BertForQuestionAnswering, BertTokenizer




class LoadQAModel:
    def __init__(self):
        self.QA_PATH = "bert-large-uncased-whole-word-masking-finetuned-squad"
        self.load_qa()


    def load_qa(self):
        self.qa_tokenizer = BertTokenizer.from_pretrained(self.QA_PATH)
        self.qa_model = BertForQuestionAnswering.from_pretrained(self.QA_PATH)

    def answer_questions(self, questions, abstract):
        answered = []

        for question in questions:
            input_ids = self.qa_tokenizer.encode(question, abstract)
            tokens = self.qa_tokenizer.convert_ids_to_tokens(input_ids)

            sep_index = input_ids.index(self.qa_tokenizer.sep_token_id)
            num_seg_a = sep_index + 1
            num_seg_b = len(input_ids) - num_seg_a
            segment_ids = [0] * num_seg_a + [1] * num_seg_b

            outputs = self.qa_model(
                torch.tensor([input_ids]),
                token_type_ids=torch.tensor([segment_ids]),
                return_dict=True
            )

            start_tokens = outputs.start_logits
            end_tokens = outputs.end_logits

            start_pos = torch.argmax(start_tokens)
            end_pos = torch.argmax(end_tokens)

            answer = tokens[start_pos]

            for i in range(start_pos + 1, end_pos + 1):
                if tokens[i][0:2] == '##':
                    answer += tokens[i][2:]

                else:
                    answer += ' ' + tokens[i]

            answered += [answer]

        return answered