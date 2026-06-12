from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
import torch
import pandas as pd
import numpy as np
import time
#
######################PER STEP LATENCIES/THROUGHPUTS ON A DISTRIBUTED SETUP############

latencies = []
throughputs = []
#
class Benchmarking(TrainerCallback):
    pass
'''
These are only for training/fine-tuning not for evaluation
    def on_epoch_begin(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    def on_epoch_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        epoch_duration = time.perf_counter() - self.start_time
        #print(f'Time elapsed in epoch(epoch latency) {state.epoch}: {epoch_duration}')
    def on_step_begin(self,args,state,control,**kwargs):
      torch.cuda.synchronize()#pseftika parapano
      self.step_start_time = time.perf_counter()
    def on_step_end(self,args,state,control,**kwargs):
      torch.cuda.synchronize() #perimene na teliosoun oi gpus tous ypologismous gia na mhn stamatiseis na metras eno borei na trexoun akoma oi gpu
      step_latency = time.perf_counter() - self.step_start_time
      #print(f'Step {state.global_step} Latency: {step_latency:.2f}')
      step_throughput = args.per_device_train_batch_size / step_latency #samples in this batch that where processed in latency time
      #assuming 1 step = 1 batch else 1 step = batch_size * gradient_accumulation_steps(default 1)
      #print(f'Step {state.global_step} Throughput: {step_throughput:.2f} samples/sec')
      #mean time / batch , batch1 n1 secs,batch2 n2 secs etc
      latencies.append(step_latency)
      throughputs.append(step_throughput)
'''



#
###CUDA###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###Load Saved tokenizer-model config- and lora weights###
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizers/HDFS/distilbert')

###Load Model config and adapter weights###

config = AutoConfig.from_pretrained("/storage/data2/up1072604/saved_models/HDFS/distilbert")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',config=config)
#
lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_models/HDFS/distilbert')
lora = lora.merge_and_unload()
lora = lora.to(device)
##
print(lora.config.id2label)
print(lora.config.label2id)
print('Num labels:',lora.config.num_labels)
##
lora.eval() ###Evaluation mode since we are running inference

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


###FOCAL LOSS FUNCTION###
class SparseCategoricalFocalLoss(nn.Module):
  def __init__(self,gamma=2,alpha=None,reduction='mean'):
    super().__init__() 
    self.gamma = gamma
    self.reduction = reduction
    self.alpha = alpha
  def forward(self,logits,labels):
    self.alpha = self.alpha.to(device)
    propabilities = F.softmax(logits,dim=-1) #propabilities(logits to probs with softmax)
    #dimensions (batch,no_of_classes)-eg.(batch,2)
    ##(batch,1)
    labels = labels.view(-1,1)
    #(batch,)
    true_propabilities = propabilities.gather(1, labels).squeeze(1)
    #
    alpha_factor = self.alpha.gather(0,labels.view(-1))
    #
    loss = -alpha_factor * ((1-true_propabilities)**self.gamma) * torch.log(true_propabilities + 1e-8)
    #
    return loss.mean() if self.reduction == 'mean' else loss.sum()

###TRAINER TO INCORPORATE CUSTOM LOSS FUNCTION###
class ImbalancedTrainer(Trainer):
	def __init__(self,*args,loss_fn=None,**kwargs):
		super().__init__(*args,**kwargs)
		self.loss_fn = SparseCategoricalFocalLoss(gamma=2,alpha=weights,reduction='mean')
	def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
		labels = inputs.pop('labels') #Get ground truth(expected output)
		outputs = model(**inputs)
		logits = outputs.get('logits') #get the model's output(logits) for these inputs
		#compute loss difference between logits and expected output
		loss = self.loss_fn(logits,labels)
		#
		return (loss,outputs) if return_outputs else loss
        
###################################
training_arguments = TrainingArguments(
    output_dir = '/storage/data2/up1072604/run', #Location where the fine tuned model's weights will be stored
    overwrite_output_dir=True,  # When fine tuning starts overwrite the above directory
    per_device_eval_batch_size=64, #batch size for evaluation
)
###Instantiate ImbalancedTrainer###
trainer = ImbalancedTrainer(
    model=lora, #The model
    args=training_arguments, #Training arguments
    compute_metrics=None,
    callbacks = [Benchmarking()]
 )

###########################--------INFERENCE/EVALUATING ON THE WHOLE  SET ##########
latencies = []
torch.cuda.synchronize() #leftover gpu work from earlier
start = time.perf_counter()
#SINCE WE ARE EVALUATING BENCHMARKS NO ACCURACY,PRECISION,RECALL METRICS ARE COMPUTED. BENCHMARKS ARE
#EVALUATED ON THE WHOLE DATASET AS A CEILING FOR THIS PROBLEM. SINCE IT IS BENCHMARKING IT DOESNT
#MATTER THAT THE MODEL HAS ALREAD SEEN SOME OF THE DATA IT DOES NOT AFFECT THE SPEED, LATENCY,THROUGHPUT OF THE MODEL
eval_results = trainer.evaluate(eval_dataset=event_traces)
torch.cuda.synchronize()
if trainer.is_world_process_zero():
    print(f'WALL CLOCK INFERENCE/EVALUATION TIME :{time.perf_counter() - start:.2f}') #### STOP COUNTING AFTER EVALUATION
    print(f"Global Latency(Trainer): {eval_results.metrics['eval_runtime']:.2f}")
    print(f"Global Throughput defined as N_samples/time{len(event_traces)} it took for these samples: {len(event_traces)/eval_results.metrics['eval_runtime']:.2f}")
    print(f"Global Throughput(Trainer): {eval_results.metrics['eval_samples_per_second']:.2f}")

##
######AVERAGE STEP METRICS OTHER FILE##############
