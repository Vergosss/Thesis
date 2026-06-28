import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import time
import sys
#
Processes = int(sys.argv[1])
print(f'Number of Processes: {Processes}')
#
start = time.perf_counter()
#
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizers/HDFS/distilbert')
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
  	tokens = tokenizer(entry['text'],padding='max_length',truncation=True,max_length=512)
  	tokens['labels'] = entry['label']
  	return tokens


#Print a trace i.e a vector of events
print(event_traces.sample(1))
event_traces['text'] = event_traces.apply(features_to_strings,axis=1)
print(event_traces.sample(1))
event_traces = Dataset.from_pandas(event_traces)
event_traces = event_traces.map(tokenize_logs,batched=True,num_proc=Processes,load_from_cache_file=False)
event_traces.save_to_disk(f'/storage/data2/up1072604/data/tokenized_HDFS_{Processes}_Processes')
end = time.perf_counter()
print(f'Time for preprocessing (Data Loading,conversion,tokenizing and saving) - Preprocessing Latency: {end-start:.2f}') 
print(f'How many samples per second can this pipeline handle - Preprocessing Throughput: {len(event_traces)/(end-start):.2f}')
