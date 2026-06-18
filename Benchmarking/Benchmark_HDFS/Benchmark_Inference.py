
from transformers import AutoModelForSequenceClassification, AutoConfig,TrainingArguments, Trainer
from datasets import load_from_disk
from peft import PeftModel, PeftConfig
import torch
import pandas as pd
import numpy as np
import time
import math
import os
num_gpus= torch.cuda.device_count()
print('GPUs:',num_gpus)
#
######################PER STEP LATENCIES/THROUGHPUTS ON A DISTRIBUTED SETUP############
event_traces = load_from_disk('/storage/data2/up1072604/data/tokenized_dataset')
###Load Saved tokenizer-model config- and lora weights###

###Load Model config and adapter weights###

config = AutoConfig.from_pretrained("/storage/data2/up1072604/saved_models/HDFS/distilbert")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',config=config)
#

lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_models/HDFS/distilbert')
lora = lora.merge_and_unload()
'''
print('RANKS',os.environ["RANK"])
print('WORLD_SIZE',os.environ["WORLD_SIZE"])
#lora = lora.to(device)
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

lora = lora.to(f"cuda:{local_rank}")
'''
print("WHERE IT LIVES?:", next(lora.parameters()).device)

##
print(lora.config.id2label)
print(lora.config.label2id)
print('Num labels:',lora.config.num_labels)
##
lora.eval() ###Evaluation mode since we are running inference
	###Not really need since After we previously fine-tuned with lora the config it saves has already inference_mode=True
	##########Load Data##########
	#Create text from vector -> convert to dataset -> tokenize texts

 
###################################
training_arguments = TrainingArguments(
    output_dir = '/storage/data2/up1072604/run', #Location where the fine tuned model's weights will be stored
    overwrite_output_dir=True,  # When fine tuning starts overwrite the above directory
    per_device_eval_batch_size=128, #batch size for evaluation
    dataloader_num_workers=4,
    dataloader_pin_memory=True
)
##################Remove loss computation entirely#########
'''
class ImbalancedTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
        return (outputs, None, None)
###Instantiate ImbalancedTrainer###
'''
trainer = Trainer(
    model=lora, #The model
    args=training_arguments, #Training arguments
    compute_metrics=None
)
print("WHERE IT LIVES?:", next(lora.parameters()).device)
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
    print(f"Number of processes: {trainer.args.world_size}, Number of GPUs: {num_gpus} ,Batch size per gpu: {trainer.args.per_device_eval_batch_size}, Number of global steps: {len(trainer.get_eval_dataloader(event_traces))} vs Manually computed : {math.ceil(len(event_traces)/(num_gpus*trainer.args.per_device_eval_batch_size))}")
    #####################################GLOBAL BENCHMARKS(WALL CLOCK TIMES
    print(f'WALL CLOCK INFERENCE/EVALUATION TIME :{time.perf_counter() - start:.2f}') #### STOP COUNTING AFTER EVALUATION
    print(f"Global Latency(Trainer): {eval_results['eval_runtime']:.2f}")
    print(f"Global Throughput defined as N_samples/time {len(event_traces)} samples it took for these samples: {len(event_traces)/eval_results['eval_runtime']:.2f}")
    print(f"Global Throughput(Trainer): {eval_results['eval_samples_per_second']:.2f}")
    ######AVERAGE STEP METRICS##############
    ##########For these N steps it took total wall clock time to process them.
    #########Total steps
    print(f"Average Step Latency(Total time divided by number of steps): {eval_results['eval_runtime']/math.ceil(len(event_traces)/(num_gpus*trainer.args.per_device_eval_batch_size)):.2f}")
    ###SANITY CHECK################
    print(f"Average Step Latency(Total time divided by number of steps): {eval_results['eval_runtime']/len(trainer.get_eval_dataloader(event_traces)):.2f}")

