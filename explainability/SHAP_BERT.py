from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel, PeftConfig
import pandas as pd
import numpy as np
import torch
import shap
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###
class CustomTokenizer:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self,text, *args, **kwargs):
        kwargs["truncation"] = True
        #kwargs["padding"] =  "max_length"
        kwargs["max_length"] =  self.max_length
        return self.tokenizer(text,*args, **kwargs)
    def encode(self, text, *args, **kwargs):
        kwargs["truncation"] = True
        kwargs["max_length"] = self.max_length
        return self.tokenizer.encode(text, *args, **kwargs)

    def encode_plus(self, text, *args, **kwargs):
        kwargs["truncation"] = True
        kwargs["max_length"] = self.max_length
        return self.tokenizer.encode_plus(text, *args, **kwargs)

    def batch_encode_plus(self, batch_texts, *args, **kwargs):
        kwargs["truncation"] = True
        kwargs["max_length"] = self.max_length
        return self.tokenizer.batch_encode_plus(batch_texts, *args, **kwargs)
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
###
config = AutoConfig.from_pretrained("/storage/data2/up1072604/saved_models/HDFS/distilbert")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',config=config)
###
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizers/HDFS/distilbert')
tokenizer = CustomTokenizer(tokenizer)
###
lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_models/HDFS/distilbert')
lora = lora.merge_and_unload()
lora = lora.to(device)
######
print(lora.config.id2label)
print(lora.config.label2id)
print('Num labels:',lora.config.num_labels)
lora.eval()
###
event_traces = pd.read_csv('/storage/data2/up1072604/data/Event_traces.csv',usecols=['BlockId','Label','Features'])
event_traces['Label'] = event_traces['Label'].map({'Success':0,'Fail':1})
###Get the templates to match with###
log_templates = pd.read_csv('/storage/data2/up1072604/data/HDFS.log_templates.csv')
###Drop Block Id###
event_traces.drop(columns=['BlockId'],inplace=True) #drop the block id
###
event_traces.rename(columns={'Features':'text','Label':'label'}, inplace=True) ###rename features to text
event_traces.info()
###Dictionary of EventIds-Event text###
event_dictionary = dict(zip(log_templates['EventId'],log_templates['EventTemplate']))
###Train-val-test split- Get the test set the same set from training script by setting same random state###
event_traces_train,event_traces_test = train_test_split(event_traces,test_size=0.1,stratify=event_traces['label'],random_state=42,shuffle=True)
event_traces_train,event_traces_validation = train_test_split(event_traces_train,test_size=0.1111,stratify=event_traces_train['label'],random_state=42,shuffle=True)
###Sample from the test set###
###Apply on each row of the dataset###
def features_to_strings(entry):
  return " ".join([event_dictionary.get(eventID) for eventID in entry['text'].replace('[','').replace(']','').split(',')])
##
event_traces_train['text'] = event_traces_train.apply(features_to_strings,axis=1)
event_traces_validation['text'] = event_traces_validation.apply(features_to_strings,axis=1)
event_traces_test['text'] = event_traces_test.apply(features_to_strings,axis=1)
print(event_traces_train.sample(1))
###Subsample for display###
event_traces_test_1 = event_traces_test[event_traces_test['label'] == 1].sample(n=5,random_state=42)
event_traces_test_0 = event_traces_test[event_traces_test['label'] == 0].sample(n=5,random_state=42)
event_traces_test = pd.concat([event_traces_test_0,event_traces_test_1])
###Shuffle###
event_traces_test = event_traces_test.sample(frac=1,random_state=42)
print('Counts:',event_traces_test['label'].value_counts())
event_traces_test = Dataset.from_pandas(event_traces_test)
###Explaining##

explanations = []
#attributions = []

##############PREDICTION FUNCTION#######################
explainer = shap.Explainer(,tokenizer)
shaps = explainer(event_traces_test)
html = shaps.plot.text(shaps)
with open('/storage/data2/up1072604/saves/explanations_shap.html','w') as file:
  file.writelines(explanations)
'''
print(attributions)
for attr in attributions:
    for _,score in attr:
        print(score)
attributions = [[score for _,score in attr] for attr in attributions]
'''
