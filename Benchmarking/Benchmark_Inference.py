from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#

latencies = []
throughputs = []
#
class Benchmarking_per_Epoch(TrainerCallback):

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


#
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
lora.eval() ###Evaluation mode since we are running inference

####################

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

######################
###Number of distinct labels in dataset###
no_of_labels = int(event_traces['Label'].nunique())
###Calculate class weights with the inverse class frequency(inverse of each class percentage in the train dataset)##
weights = event_traces_train['Label'].value_counts(normalize=True)
weights = torch.tensor([1/weights.loc[x] for x in sorted(list(weights.index))])
print(weights)
##################
###Convert to Huggingface Dataset###
event_traces_train = Dataset.from_pandas(event_traces_train)
event_traces_test = Dataset.from_pandas(event_traces_test)
event_traces_validation = Dataset.from_pandas(event_traces_validation)

##############
###Tokenizing###
event_traces_train = event_traces_train.map(tokenize_logs,batched=True)
event_traces_test = event_traces_test.map(tokenize_logs,batched=True)
event_traces_validation = event_traces_validation.map(tokenize_logs,batched=True)
###########################

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
    eval_strategy = "epoch", #Evaluation should be done at the end of each epoch
    learning_rate=2e-5, #small learning rate -> better generalization
    per_device_train_batch_size=16, #batch size for the training set
    per_device_eval_batch_size=64, #batch size for evaluation
    num_train_epochs=3, #epochs for the model to run
    weight_decay=0.01, #Regularization to reduce overfitting
    save_strategy= "no" #Don't save checkpoints
)
###Instantiate ImbalancedTrainer###
trainer = ImbalancedTrainer(
    model=lora, #The model
    args=training_arguments, #Training arguments
 )

results = trainer.evaluate(eval_dataset=event_traces_test) #Evaluate on unseen test subset
###########################--------INFERENCE/EVALUATING ON A TEST SET ##########
latencies = []
torch.cuda.synchronize() #leftover gpu work from earlier
start = time.perf_counter()
eval_results = trainer.evaluate(eval_dataset=event_traces_test)
torch.cuda.synchronize()
print(f'WALL CLOCK INFERENCE/EVALUATION TIME :{time.perf_counter() - start:.2f}') #### STOP COUNTING AFTER EVALUATION
print(f'Global Latency(Trainer): {results.metrics['eval_runtime']:.2f}')
print(f'Global Throughput defined as N_samples/time it took for these samples: {event_traces_test)/results.metrics['eval_runtime']:.2f}')
print(f'Global Throughput(Trainer): {results.metrics['eval_samples_per_second']:.2f}')

##
input('WAIT')
######Average batch metrics##############

print(f'Average Batch/step latency: {sum(latencies)/len(latencies):.2f} seconds per batch/step')
print(f'Average Batch/step throughput: {sum(throughputs)/len(throughputs):.2f} samples per second')
#######Global Trainer metrics#####
print(f'Global Throughput(Trainer): {results.metrics['eval_samples_per_second']:.2f} vs Mean of Throughputs: {sum(throughputs)/len(throughputs):.2f}') #Close due to low latencies variance
print(f'Global Latency(Trainer): {results.metrics['eval_runtime']:.2f} vs Sum of Latencies: {sum(latencies):.2f}')
########Global Metrics ################
print(f'Global Throughput(total samples/trainer runtime): {len(event_traces_test)/results.metrics['eval_runtime']:.2f}')