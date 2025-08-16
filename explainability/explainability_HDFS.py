from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
from ferret import Benchmark, IntegratedGradientExplainer, SHAPExplainer, BaseDataset
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from tqdm import tqdm
#############
TEST_SET = "TEST_SET"
###Custom tokenizer because truncating behavior is not stored in the tokenizer's config###
class CustomTokenizer:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, *args, **kwargs):
        kwargs["truncation"] = True
        #kwargs["padding"] =  "max_length"
        kwargs["max_length"] =  self.max_length
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


###Custom Dataset(extending the BaseDataset class) so we can use the evaluate_samples function
class CustomDataset(BaseDataset):
    def __init__(self,dataframe):
        self.dataframe = dataframe
    def __getitem__(self,idx):
        return self.dataframe.loc[idx]
    def __len__(self):
      return len(self.dataframe)
    @property
    def NAME(self):
        return 'Custom'
    def get_instance(self, idx: int, split_type: str = TEST_SET):
        pass
    def _get_item(self, idx: int, split_type: str = TEST_SET):
        pass
    def _get_text(self, idx, split_type: str = TEST_SET):
        pass
    def _get_rationale(self, idx, split_type: str = TEST_SET):
        pass
    def _get_ground_truth(self, idx, split_type: str = TEST_SET):
        pass
    @property
    def avg_rationale_size(self):
        # Default value
        return 5
        
#############
###CUDA###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###Load Saved tokenizer-model config- and lora weights###
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizers/HDFS/roberta')

tokenizer = CustomTokenizer(tokenizer)
###Load Model config and adapter weights###

config = AutoConfig.from_pretrained("/storage/data2/up1072604/saved_models/HDFS/roberta")
model = AutoModelForSequenceClassification.from_pretrained('roberta-base',config=config)
#
lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_models/HDFS/roberta')
lora = lora.merge_and_unload()
lora = lora.to(device)
##
print(lora.config.id2label)
print(lora.config.label2id)
print('Num labels:',lora.config.num_labels)
##
lora.eval() ###Evaluation mode since we are running inference/explainability
#
###Explainers###
shap = SHAPExplainer(lora,tokenizer)
integrated_gradients = IntegratedGradientExplainer(lora,tokenizer)
#Benchmark object to run explainability###
bench = Benchmark(lora,tokenizer,explainers=[shap,integrated_gradients])
###
event_traces = pd.read_csv('/storage/data2/up1072604/data/Event_traces.csv',usecols=['BlockId','Label','Features'])
print('CHECK:',event_traces.index.duplicated().any())
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
#####################
###Get all anomalies from test set because they are too few to subsample further, random sample### 
event_traces_test_1 = event_traces_test[event_traces_test['label'] == 1]
event_traces_test_0 = event_traces_test[event_traces_test['label'] == 0].sample(n=5000-len(event_traces_test_1),random_state=42)
event_traces_test = pd.concat([event_traces_test_0,event_traces_test_1])
###Shuffle###
event_traces_test = event_traces_test.sample(frac=1,random_state=42)
####################
print('Length of subset:',len(event_traces_test))
print('Label counts:',event_traces_test['label'].value_counts(normalize=True))
print('CHECK:',event_traces_test.index.duplicated().any())
sample = list(event_traces_test.index)
###Convert to dataset- Get the indexes we want to explain###
event_traces_test_dataset = CustomDataset(event_traces_test)
###Evaluate samples###
results = bench.evaluate_samples(event_traces_test_dataset,sample=sample,target=None,show_progress_bar=True) #if target=None it will compute attributions for the predicted class
###Returns averaged metrics across all samples as defined above###
results_df = bench.show_samples_evaluation_table(results,apply_style=False) #Return raw dataframe not styled/jupyter dependent
print(results_df)
