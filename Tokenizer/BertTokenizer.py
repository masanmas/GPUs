from transformers import BertTokenizer

TOKENIZE_CASED = BertTokenizer.from_pretrained("bert-base-cased")
TOKENIZE_UNCASED = BertTokenizer.from_pretrained("bert-base-uncased")

RES_CASED = TOKENIZE_CASED.tokenize("This is a Bert test GPU!, I'm going to see what this gets as a result.")
RES_UNCASED = TOKENIZE_UNCASED.tokenize("This is a Bert test GPU!, I'm going to see what this gets as a result.")


print("CASED -------------- \n", RES_CASED)
print("UNCASED --------------- \n", RES_UNCASED)

ACCENTS_CASED = TOKENIZE_CASED.tokenize("Sí, dijo el chef.")
ACCENTS_UNCASED = TOKENIZE_UNCASED.tokenize("Sí, dijo el chef.")

print("CASED --------------- \n", ACCENTS_CASED)
print("UNCASED ---------------- \n", ACCENTS_UNCASED)

