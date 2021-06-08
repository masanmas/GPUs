import torch
from transformers import BertModel, BertTokenizer, BertForMaskedLM

class LoadMaskModel:
    def __init__(self):
        self.MASK_PATH = "bert-base-multilingual-cased"
        self.load_mask()

    #MASK UTILIDAD
    def load_mask(self):
        self.mask_tokenizer = BertTokenizer.from_pretrained(self.MASK_PATH)
        self.mask_model = BertForMaskedLM.from_pretrained(self.MASK_PATH)

        self.mask_model.eval()

    def tokenize_mask(self, fraseEntrada):
        maskEntradaTokens = self.mask_tokenizer.tokenize(fraseEntrada)
        maskEntradaIDs = self.mask_tokenizer.convert_tokens_to_ids(maskEntradaTokens)

        maskTokensTensor = torch.tensor([maskEntradaIDs])
        segmentTokens = [0] * len(maskEntradaTokens)
        segmentTokens = torch.tensor([segmentTokens])

        return {
            'TOKENS': maskTokensTensor,
            'SEGMENTS': segmentTokens
        }

    def evaluate_mask(self, fraseEntrada, maskPos):
        maskData = self.tokenize_mask(fraseEntrada=fraseEntrada)

        with torch.no_grad():
            maskOutputs = self.mask_model(maskData['TOKENS'], maskData['SEGMENTS'])

        predicted_index = torch.argmax(maskOutputs[0][0][maskPos]).item()
        predicted_token = self.mask_tokenizer.convert_ids_to_tokens([predicted_index])[0]

        return predicted_token
