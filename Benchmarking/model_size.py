from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel, PeftConfig
import torch
import numpy as np
import os
num_gpus= torch.cuda.device_count()
print('GPUs:',num_gpus)
#

###Load Model config and adapter weights. Merge into one model the merged model will be used for inference benchmarking###

config = AutoConfig.from_pretrained("/storage/data2/up1072604/saved_models/HDFS/distilbert")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',config=config)
#
lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_models/HDFS/distilbert')
lora = lora.merge_and_unload()
###############################
precision = 4 #in bytes
print(f'Model\'s Numerical Precision {set(parameter.dtype for parameter in model.parameters())}')
number_of_parameters = sum(parameter.numel() for parameter in lora.parameters()) #since we are doing inference all parameters are used. If it was fine tuning then requires_grad
number_of_buffers = sum(buffer.numel() for buffer in lora.buffers())
print(f'Model size in MegaBytes: {(number_of_parameters * precision + number_of_buffers* precision) / (1024**2) :.2f} ')
