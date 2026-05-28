from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import pandas as pd
import numpy as np
import torch
from transformers_interpret import SequenceClassificationExplainer
###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequences_test = pd.read_csv('/storage/data2/up1072604/saves/sequences_test.csv')
###
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizer')
lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_model')
lora = lora.merge_and_unload()
lora = lora.to(device)
######
lora.eval()
explainer = SequenceClassificationExplainer(lora, tokenizer)
explanations = []
attributions = []
#Ta attributions tou IG einai ena list/array me tuples (token,score)##
def explain(entry):
    attribution = explainer(entry['text'],class_name="")
    attributions.append(attribution)
    html = explainer_IG.visualize()
    explanations.append(f'html.data\n')
#############
input('WAIT')
sequences.apply(explain,axis=1)
print(attributions)
for attr in attributions:
    for _,score in attr:
        print(score)
attributions = [[score for _,score in attr] for attr in attributions]
