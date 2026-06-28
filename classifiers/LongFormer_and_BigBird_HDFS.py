import pandas as pd
import numpy as np
from transformers import AutoTokenizer, LongformerForSequenceClassification, BigBirdForSequenceClassification,Trainer,TrainingArguments,TrainingArguments,TrainerCallback,DataCollatorWithPadding
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset,load_from_disk
import evaluate
from collections import Counter
#################Big Bird or Longformer-Tokenizer and Collator###############
tokenizer_name = "google/bigbird-roberta-base"
#"google/bigbird-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,max_length=4096)
print(tokenizer.model_max_length)
collator = DataCollatorWithPadding(tokenizer)
###Get the graphics card###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
################LOAD DATA###############
event_traces_train = load_from_disk('/storage/data2/up1072604/data/tokenized_HDFS_train_bigbird')
event_traces_validation = load_from_disk('/storage/data2/up1072604/data/tokenized_HDFS_validation_bigbird')
event_traces_test = load_from_disk('/storage/data2/up1072604/data/tokenized_HDFS_test_bigbird')
##Describe the Dataset###
print('Event traces train huggingface dataset:',event_traces_train)
###Verify Distribution of Labels in subsets###


###A Random Sample of train subset to verify everything is ok###
print('Random train sample',event_traces_train.shuffle(seed=42).select(range(1)))

###Number of distinct labels in dataset###
counter = Counter(event_traces_train['labels'])
no_of_labels = int(len(counter))
print('No of labels:',no_of_labels)
###Calculate class weights with the inverse class frequency(inverse of each class percentage in the train dataset)##

weights = torch.tensor([counter.total()/counter[x] for x in sorted(list(counter.keys()))]) #simpler: for x in sorted(list(counter)) #it is 1/counter[x]/counter.total()
print('Weights vector:',weights)

#input('WAIT')
###tokenizer and relative function###
#tokenizer_name = "google/bigbird-roberta-base"
#"allenai/longformer-base-4096"
###LoRa Config###
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, #Task type. We classify texts so sequence classification
    r=7, #Common dimension between A and B matrices
    lora_alpha=14, #Alpha hyperparameter -> usually 2*r
    lora_dropout=0.1,
    inference_mode = False,
    target_modules=["query", "key","value"] #For BERT,RoBERTa,ALBERT,Distilroberta,BigBird
    #target_modules = ["query", "key","value","query_global","key_global","value_global"] # for , Longformer only
   #target_modules = ["q_lin","v_lin","k_lin"] #For DistilBERT
)
###MODEL###
ground_truth = ['Benign','Anomaly']
label2id = {label:id for id,label in enumerate(ground_truth)}
id2label = {id:label for id,label in enumerate(ground_truth)}
#

#longformer = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096",num_labels=no_of_labels,id2label=id2label,label2id=label2id,gradient_checkpointing=True)
#print('Longformer sliding window: ',longformer.config.attention_window)
bigbird = BigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-base",num_labels=no_of_labels,id2label=id2label,label2id=label2id,gradient_checkpointing=True)
print('BigBird sliding window: ',bigbird.config.block_size)

#Encapsulate
lora = get_peft_model(bigbird,lora_config)
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
  matrix.to_csv('/storage/data2/up1072604/saves/HDFS/bigbird/bigbird_confusion.csv')
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
    per_device_train_batch_size=4, #batch size for the training set
    per_device_eval_batch_size=8, #batch size for evaluation
    num_train_epochs=3, #epochs for the model to run
    weight_decay=0.01, #Regularization to reduce overfitting
    save_strategy= "no", #Don't save checkpoints
    gradient_checkpointing=True,
    gradient_accumulation_steps=8,
    bf16=True,
    group_by_length=True
)
###Instantiate ImbalancedTrainer###
trainer = ImbalancedTrainer(
    model=lora, #The model
    args=training_arguments, #Training arguments
    train_dataset=event_traces_train, #Training set
    eval_dataset=event_traces_validation, # validation to set on this the model will be evaluated at the end of each epoch
    compute_metrics=compute_metrics, #Evaluation function to run at each epoch
    processing_class=tokenizer,
    data_collator=collator
   )
#input('WAIT')
###Train/Fine-tune the model###
trainer.train()
###Change Evaluation function to calculate confusion matrix- Evaluation###
trainer.compute_metrics = compute_metrics_test
results = trainer.evaluate(eval_dataset=event_traces_test) #Evaluate on unseen test subset
print(results)

###Save the model###
tokenizer.save_pretrained('/storage/data2/up1072604/saved_tokenizers/HDFS/bigbird') #save the tokenizer
bigbird.config.save_pretrained('/storage/data2/up1072604/saved_models/HDFS/bigbird') #save the base model's config such as id2label etc
lora.save_pretrained('/storage/data2/up1072604/saved_models/HDFS/bigbird') #Save the reduced matrices
#########
#tokenizer.save_pretrained('/storage/data2/up1072604/saved_tokenizers/HDFS/bigbird') #save the tokenizer
#model.config.save_pretrained('/storage/data2/up1072604/saved_models/HDFS/bigbird') #save the base model's config such as id2label etc
#lora.save_pretrained('/storage/data2/up1072604/saved_models/HDFS/bigbird') #Save the reduced matrices
