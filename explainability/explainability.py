from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
from ferret import Benchmark, IntegratedGradientExplainer, SHAPExplainer, BaseDataset
import torch
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
#############
TEST_SET = "TEST_SET"
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
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizers/IoT23/binary')
#
config = AutoConfig.from_pretrained("/storage/data2/up1072604/saved_models/IoT23/binary")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',config=config)
#
lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_models/IoT23/binary')
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
#
bench = Benchmark(lora,tokenizer,explainers=[shap,integrated_gradients])
###
sequences = pd.read_csv('/storage/data2/up1072604/data/IoT23_sequences_binary.csv')
sequences = sequences.sample(frac=1,random_state=42) #Shuffle. Do this only for iot23-edgeiiotset NOT HDFS(HDFS does not shuffle at first!)
###Train-val-test split- Get the test set the same set from training script by setting same random state###
sequences_train,sequences_test = train_test_split(sequences,test_size=0.1,stratify=sequences['label'],random_state=42,shuffle=True)
sequences_train,sequences_validation = train_test_split(sequences_train,test_size=0.1111,stratify=sequences_train['label'],random_state=42,shuffle=True)
###Sample from the test set###
sequences_test = sequences_test.sample(n=1000,random_state=42)
###Convert to dataset- Get the indexes we want to explain###
sequences_test_dataset = CustomDataset(sequences_test)
sample = 
###Evaluate samples###
#sample= lista me ta indexes ton keimenon pou thelo na do
results = bench.evaluate_samples(sequences_test_dataset,sample=,target=None,show_progress_bar=True) #if target=None it will compute attributions for the predicted class
###Returns averaged metrics across all samples as defined above###
results_df = bench.show_samples_evaluation_table(results,apply_style=False) #Return raw dataframe not styled/jupyter dependent
print(results_df)
