import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification,Trainer,TrainingArguments,TrainingArguments,TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
###Get the graphics card###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
###Load event vectors/blocks###
event_traces = pd.read_csv('/storage/data2/up1072604/data/Event_traces.csv',usecols=['BlockId','Label','Features'])
event_traces['Label'] = event_traces['Label'].map({'Success':0,'Fail':1})
###Get the templates to match with###
log_templates = pd.read_csv('/storage/data2/up1072604/data/HDFS.log_templates.csv')

###Dictionary of EventIds-Event text###
event_dictionary = dict(zip(log_templates['EventId'],log_templates['EventTemplate']))

print(event_dictionary)

###Information-Dimensions-Overall Description###
event_traces.info()
print(event_traces.shape) 
print(event_traces.describe())
#
###Convert feature event vector to text seperated with space ###
def features_to_strings(entry):
  return " ".join([event_dictionary.get(eventID) for eventID in entry['Features'].replace('[','').replace(']','').split(',')])

print('Class distribution of Labels: ',event_traces['Label'].value_counts())
###Drop Unnecessary columns###
event_traces.drop(columns=['BlockId'],inplace=True) #drop the block id
###Make sure labels are integers not floats###
event_traces['Label'] = event_traces['Label'].astype(int)

###Train test dev split###
event_traces_train,event_traces_test = train_test_split(event_traces,test_size=0.1,random_state=42,stratify=event_traces['Label'],shuffle=True)
event_traces_train,event_traces_validation = train_test_split(event_traces_train,test_size=0.1111,stratify=event_traces_train['Label'],random_state=42,shuffle=True)
###Verify Distribution of Labels in subsets###
print('Class distribution train: ',event_traces_train['Label'].value_counts())
print('Class distribution validation: ',event_traces_validation['Label'].value_counts())
print('Class distribution test: ',event_traces_test['Label'].value_counts())
###Apply on each row of the dataset###
event_traces_train['Features'] = event_traces_train.apply(features_to_strings,axis=1)
event_traces_validation['Features'] = event_traces_validation.apply(features_to_strings,axis=1)
event_traces_test['Features'] = event_traces_test.apply(features_to_strings,axis=1)
###A Random Sample of train subset to verify everything is ok###
print(event_traces_train.sample(1))

###Number of distinct labels in dataset###
no_of_labels = int(event_traces['Label'].nunique())
###Calculate class weights with the inverse class frequency(inverse of each class percentage in the train dataset)##
weights = event_traces_train['Label'].value_counts(normalize=True)
weights = torch.tensor([1/weights.loc[x] for x in sorted(list(weights.index))])
print(weights)

###Convert to Huggingface Dataset###
event_traces_train = Dataset.from_pandas(event_traces_train)
event_traces_test = Dataset.from_pandas(event_traces_test)
event_traces_validation = Dataset.from_pandas(event_traces_validation)
###tokenizer and relative function###
tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
def tokenize_logs(entry):
  tokens = tokenizer(entry['Features'],padding='max_length',truncation=True)
  tokens['labels'] = entry['Label']
  return tokens

###Tokenizing###
event_traces_train = event_traces_train.map(tokenize_logs,batched=True)
event_traces_test = event_traces_test.map(tokenize_logs,batched=True)
event_traces_validation = event_traces_validation.map(tokenize_logs,batched=True)
####################

###LoRa Config###
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, #Task type. We classify texts so sequence classification
    r=7, #Common dimension between A and B matrices
    lora_alpha=14, #Alpha hyperparameter -> usually 2*r
    lora_dropout=0.1,
    inference_mode = False,
    target_modules=["query", "key","value"] #For BERT,RoBERTa,ALBERT,Distilroberta
   #target_modules = ["q_lin","v_lin","k_lin"] #For DistilBERT
)
###MODEL###
ground_truth = ['Benign','Anomaly']
label2id = {label:id for id,label in enumerate(ground_truth)}
id2label = {id:label for id,label in enumerate(ground_truth)}

model = AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels=no_of_labels,id2label=id2label,label2id=label2id)
#Encapsulate
lora = get_peft_model(model,lora_config)
###Feed model to CUDA##
lora = lora.to(device)
###Check###
print("Lora model's number of labels:",lora.config.num_labels)
print("Lora model's label2id:",lora.config.label2id)
print("Lora model's id2label:",lora.config.id2label)

###METRICS###

accuracy = evaluate.load("accuracy")
other_metrics = evaluate.combine(["precision","recall","f1"])
confusion_matrix = evaluate.load("confusion_matrix")

###Function to evaluate Metrics###
def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=-1)
  other_metrics_scores = other_metrics.compute(predictions=predictions,references=labels,average=None) #all classes
  accuracy_score = accuracy.compute(predictions=predictions,references=labels)["accuracy"]
  all_metrics = {"accuracy":accuracy_score} #initialization
  for metric in other_metrics_scores: #appending
    for entry_pos,entry in enumerate(other_metrics_scores[metric]):
      all_metrics[f"{metric}_class_{entry_pos}"] = entry
  return all_metrics
#############################
def compute_metrics_test(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=-1)
  matrix = confusion_matrix.compute(references=labels,predictions=predictions)['confusion_matrix']
  matrix = pd.DataFrame(matrix,index=ground_truth,columns=ground_truth)
  matrix.to_csv('/storage/data2/up1072604/saves/HDFS/roberta/HDFS_confusion.csv')
  other_metrics_scores = other_metrics.compute(predictions=predictions,references=labels,average=None)
  accuracy_score = accuracy.compute(predictions=predictions,references=labels)["accuracy"] 
  all_metrics = {"accuracy":accuracy_score} #initialization
  for metric in other_metrics_scores: #appending
    for entry_pos,entry in enumerate(other_metrics_scores[metric]):
      all_metrics[f"{metric}_class_{entry_pos}"] = entry
  return all_metrics
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
 
###Training arguments###
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
    train_dataset=event_traces_train, #Training set
    eval_dataset=event_traces_validation, # validation to set on this the model will be evaluated at the end of each epoch
    compute_metrics=compute_metrics #Evaluation function to run at each epoch
   )
###Train/Fine-tune the model###
trainer.train()
###Change Evaluation function to calculate confusion matrix- Evaluation###
trainer.compute_metrics = compute_metrics_test
results = trainer.evaluate(eval_dataset=event_traces_test) #Evaluate on unseen test subset
print(results)

###Save the model###
tokenizer.save_pretrained('/storage/data2/up1072604/saved_tokenizers/HDFS/roberta') #save the tokenizer
model.config.save_pretrained('/storage/data2/up1072604/saved_models/HDFS/roberta') #save the base model's config such as id2label etc
lora.save_pretrained('/storage/data2/up1072604/saved_models/HDFS/roberta') #Save the reduced matrices
