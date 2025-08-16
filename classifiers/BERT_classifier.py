import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import evaluate
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
os.environ["TOKENIZERS_PARALLELISM"] = "false"
###Get the graphics card###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
###Load Data###
sequences = pd.read_csv('/storage/data2/up1072604/data/IoT23_sequences_multi.csv')
###shuffle###
sequences = sequences.sample(frac=1,random_state=42)

###Make sure labels are integers not floats###
sequences['label'] = sequences['label'].astype(int)
###Information-Dimensions-Columns-Label Distribution-Unique labels###

sequences.info()
print('Dimensions: ',sequences.shape)
print('Columns: ',sequences.columns)
print('Label distribution:',sequences['label'].value_counts())
print('Unique labels: ',sequences['label'].unique())

###Train-val-test split###
sequences_train,sequences_test = train_test_split(sequences,test_size=0.1,stratify=sequences['label'],random_state=42,shuffle=True)
sequences_train,sequences_validation = train_test_split(sequences_train,test_size=0.1111,stratify=sequences_train['label'],random_state=42,shuffle=True)
#####
print('Train: ',sequences_train['label'].value_counts(normalize=True)*100)
print('Validation: ',sequences_validation['label'].value_counts(normalize=True)*100)
print('Test: ',sequences_test['label'].value_counts(normalize=True)*100)
###Number of distinct labels in dataset###
no_of_labels = int(sequences['label'].nunique())
###Calculate class weights with the inverse class frequency(inverse of each class percentage in the train dataset)###
weights = sequences_train['label'].value_counts(normalize=True)
weights = torch.tensor([1/weights.loc[x] for x in sorted(list(weights.index))])
print(weights)

###Dataframe to HuggingFace Dataset###

sequences_train = Dataset.from_pandas(sequences_train)
sequences_test = Dataset.from_pandas(sequences_test)
sequences_validation = Dataset.from_pandas(sequences_validation)

###Tokenizer###
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") 
#tokenizer.padding_side = "left"

def tokenize_logs(entry):
  tokens = tokenizer(entry['text'],padding='max_length',truncation=True)
  tokens['labels'] = entry['label']
  return tokens
  
###TOKENIZING SUB DATASETS###

sequences_train = sequences_train.map(tokenize_logs,batched=True)
sequences_test = sequences_test.map(tokenize_logs,batched=True)
sequences_validation = sequences_validation.map(tokenize_logs,batched=True)

###LoRa Config###
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, #Task type. We classify texts so sequence classification
    r=7, #Common dimension between A and B matrices
    lora_alpha=14, #Alpha hyperparameter -> usually 2*r
    lora_dropout=0.1,
    inference_mode=False,
   # target_modules=["query", "key","value"] #For BERT,RoBERTa,ALBERT,Distilroberta
   target_modules = ["q_lin","v_lin","k_lin"] #For DistilBERT
)
###MODEL###

ground_truth = ['Benign','DDoS','Okiru','PartOfAHorizontalPortScan','C&C(Command and Control)']
label2id = {label:id for id,label in enumerate(ground_truth)}
id2label = {id:label for id,label in enumerate(ground_truth)}

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=no_of_labels,id2label=id2label,label2id=label2id) 
###Encapsulate###
lora = get_peft_model(model,lora_config)
###Feed model to CUDA###
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
###
def compute_metrics_test(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=-1)
  matrix = confusion_matrix.compute(references=labels,predictions=predictions)['confusion_matrix']
  matrix = pd.DataFrame(matrix,index=ground_truth,columns=ground_truth)
  matrix.to_csv('/storage/data2/up1072604/saves/IoT23/multi/IoT23_confusion_multi.csv')
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
    propabilities = F.softmax(logits,dim=-1) #logits to propabilities
    #dimensions (batch,no_of_classes)-eg.(batch,2)
    ##from shape: (batch,1)
    labels = labels.view(-1,1)
    #to shape: (batch,)
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
    num_train_epochs=2, #epochs for the model to run
    weight_decay=0.01, #Regularization to reduce overfitting
    save_strategy= "no" #Don't save checkpoints
)
###Instantiate ImbalancedTrainer###
trainer = ImbalancedTrainer(
    model=lora, #The model
    args=training_arguments, #Training arguments
    train_dataset=sequences_train, #Training set
    eval_dataset=sequences_validation, # validation to set on this the model will be evaluated at the end of each epoch
    compute_metrics=compute_metrics #Evaluation function to run at each epoch
   )
###Train/Fine-tune the model###
trainer.train()
###Change Evaluation function to calculate confusion matrix- Evaluation###
trainer.compute_metrics = compute_metrics_test
results = trainer.evaluate(eval_dataset=sequences_test) #Evaluate on unseen test subset
print(results)
###Save the model###
tokenizer.save_pretrained('/storage/data2/up1072604/saved_tokenizers/IoT23/multi') #save the tokenizer
model.config.save_pretrained('/storage/data2/up1072604/saved_models/IoT23/multi') #save the base model's config such as id2label etc
lora.save_pretrained('/storage/data2/up1072604/saved_models/IoT23/multi') #Save the reduced matrices
#######
