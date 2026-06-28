import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import time
import numpy as np
import sys
#
Processes = int(sys.argv[1])
print(f'Number of Processes: {Processes}')
tokenizer_name = "google/bigbird-roberta-base"
#"google/bigbird-roberta-base"
start = time.perf_counter()
#

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,max_length=4096)
#
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

###Apply on each row of the dataset###
def features_to_strings(entry):
	return " ".join([event_dictionary.get(eventID) for eventID in entry['text'].replace('[','').replace(']','').split(',')])

	####Tokenizing function###
def tokenize_logs(entry):
    tokens = tokenizer(entry['text'],truncation=True,max_length=4096,padding=False) #basically dont pad let the datacollator do the paddding
    tokens['labels'] = entry['label']
    return tokens

##################


###Train test dev split###
event_traces_train,event_traces_test = train_test_split(event_traces,test_size=0.1,random_state=42,stratify=event_traces['label'],shuffle=True)
event_traces_train,event_traces_validation = train_test_split(event_traces_train,test_size=0.1111,stratify=event_traces_train['label'],random_state=42,shuffle=True)

#Print a trace i.e a vector of events
print(event_traces_train.sample(1))
event_traces_train['text'] = event_traces_train.apply(features_to_strings,axis=1)
event_traces_validation['text'] = event_traces_validation.apply(features_to_strings,axis=1)
event_traces_test['text'] = event_traces_test.apply(features_to_strings,axis=1)

print(event_traces_train.sample(1))
####

###Convert to Huggingface Dataset###
event_traces_train = Dataset.from_pandas(event_traces_train)
event_traces_validation = Dataset.from_pandas(event_traces_validation)
event_traces_test = Dataset.from_pandas(event_traces_test)

#
event_traces_train = event_traces_train.map(tokenize_logs,batched=True,num_proc=Processes,load_from_cache_file=False)
event_traces_validation = event_traces_validation.map(tokenize_logs,batched=True,num_proc=Processes,load_from_cache_file=False)
event_traces_test = event_traces_test.map(tokenize_logs,batched=True,num_proc=Processes,load_from_cache_file=False)
####################
####Save to Disk for reuse############
event_traces_train.save_to_disk(f'/storage/data2/up1072604/data/tokenized_HDFS_train_bigbird')
event_traces_validation.save_to_disk(f'/storage/data2/up1072604/data/tokenized_HDFS_validation_bigbird')
event_traces_test.save_to_disk(f'/storage/data2/up1072604/data/tokenized_HDFS_test_bigbird')
####
end = time.perf_counter()
print(f'Time for preprocessing (Data Loading,conversion,tokenizing and saving) - Preprocessing Latency: {end-start:.2f}') 
print(f'How many samples per second can this pipeline handle - Preprocessing Throughput: {len(event_traces)/(end-start):.2f}')
