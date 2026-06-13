from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
import torch
import pandas as pd
import numpy as np
import time
import math
#
######################PER STEP LATENCIES/THROUGHPUTS ON A DISTRIBUTED SETUP############

###Load Saved tokenizer-model config- and lora weights###
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizers/HDFS/distilbert')

###Load Model config and adapter weights###

config = AutoConfig.from_pretrained("/storage/data2/up1072604/saved_models/HDFS/distilbert")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',config=config)
#
lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_models/HDFS/distilbert')
lora = lora.merge_and_unload()
##lora = lora.to(device)
##
print(lora.config.id2label)
print(lora.config.label2id)
print('Num labels:',lora.config.num_labels)
##
lora.eval() ###Evaluation mode since we are running inference
###Not really need since After we previously fine-tuned with lora the config it saves has already inference_mode=True
##########Load Data##########

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
##

####Tokenizing function###
def tokenize_logs(entry):
  tokens = tokenizer(entry['text'],padding='max_length',truncation=True)
  tokens['labels'] = entry['label']
  return tokens


#Print a trace i.e a vector of events
print(event_traces.sample(1))

######################
###Number of distinct labels in dataset###
no_of_labels = int(event_traces['Label'].nunique())
###Calculate class weights with the inverse class frequency(inverse of each class percentage in the dataset)##
weights = event_traces['Label'].value_counts(normalize=True) #same as if i had splittd because the split was stratified, percentage would be the same
weights = torch.tensor([1/weights.loc[x] for x in sorted(list(weights.index))])
print(weights)
##################
#Create text from vector -> convert to dataset -> tokenize texts
event_traces['text'] = event_traces.apply(features_to_strings,axis=1)
event_traces = Dataset.from_pandas(event_traces)
event_traces = event_traces.map(tokenize_logs,batched=True)
 
###################################
training_arguments = TrainingArguments(
    output_dir = '/storage/data2/up1072604/run', #Location where the fine tuned model's weights will be stored
    overwrite_output_dir=True,  # When fine tuning starts overwrite the above directory
    per_device_eval_batch_size=128, #batch size for evaluation
)
###Instantiate ImbalancedTrainer###
trainer = Trainer(
    model=lora, #The model
    args=training_arguments, #Training arguments
    compute_metrics=None,
    compute_loss = None
 )

###########################--------INFERENCE/EVALUATING ON THE WHOLE  SET ##########
torch.cuda.synchronize() #leftover gpu work from earlier
start = time.perf_counter()
#SINCE WE ARE EVALUATING BENCHMARKS NO ACCURACY,PRECISION,RECALL METRICS ARE COMPUTED. BENCHMARKS ARE
#EVALUATED ON THE WHOLE DATASET AS A CEILING FOR THIS PROBLEM. SINCE IT IS BENCHMARKING IT DOESNT
#MATTER THAT THE MODEL HAS ALREAD SEEN SOME OF THE DATA IT DOES NOT AFFECT THE SPEED, LATENCY,THROUGHPUT OF THE MODEL
eval_results = trainer.evaluate(eval_dataset=event_traces)
torch.cuda.synchronize()
if trainer.is_world_process_zero():
    ###################################SANITY CHECK################
    print(f"Dataset length: {len(event_traces)}")
    print(f"Number of GPUs: {trainer.args.world_size}, Batch size per gpu: {trainer.args.per_device_eval_batch_size}, Number of global steps: {len(trainer.get_eval_dataloader(event_traces))} vs Manually computed : {math.ceil(len(event_traces)/(trainer.args.world_size*trainer.args.per_device_eval_batch_size))}")
    #####################################GLOBAL BENCHMARKS(WALL CLOCK TIMES
    print(f'WALL CLOCK INFERENCE/EVALUATION TIME :{time.perf_counter() - start:.2f}') #### STOP COUNTING AFTER EVALUATION
    print(f"Global Latency(Trainer): {eval_results.metrics['eval_runtime']:.2f}")
    print(f"Global Throughput defined as N_samples/time {len(event_traces)} samples it took for these samples: {len(event_traces)/eval_results.metrics['eval_runtime']:.2f}")
    print(f"Global Throughput(Trainer): {eval_results.metrics['eval_samples_per_second']:.2f}")
    ######AVERAGE STEP METRICS##############
    ##########For these N steps it took total wall clock time to process them.
    #########Total steps
    print(f"Average Step Latency(Total time divided by number of steps): {eval_results.metrics['eval_runtime']/math.ceil(len(event_traces)/(trainer.args.world_size*trainer.args.per_device_eval_batch_size)):.2f}")
    ###SANITY CHECK################
    print(f"Average Step Latency(Total time divided by number of steps): {eval_results.metrics['eval_runtime']/len(trainer.get_eval_dataloader(event_traces)):.2f}")

