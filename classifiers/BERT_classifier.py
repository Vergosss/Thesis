#import tensorflow as tf
#from focal_loss import SparseCategoricalFocalLoss
from peft import LoraConfig, TaskType, get_peft_model
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback
from transformers_interpret import SequenceClassificationExplainer
from datasets import Dataset
import evaluate
from scipy.special import expit
import  pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
#input('WAIT')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#
print(device)
###Load Data
sequences = pd.read_csv('/storage/data2/up1072604/data/EdgeIIoT_distinct_binary_2.csv')
#shuffle
sequences = sequences.sample(frac=1,random_state=42)
#sequences['label'] = sequences['label'].apply(lambda x: 0 if x == 'Benign' else 1)
##
sequences['label'] = sequences['label'].astype(int)
sequences.info()
print('Dimensions: ',sequences.shape)
print('Columns: ',sequences.columns)
print('Label distribution:',sequences['label'].value_counts())
print('Unique labels: ',sequences['label'].unique())

###
##Train-val-test split
sequences_train,sequences_test = train_test_split(sequences,test_size=0.1,stratify=sequences['label'],random_state=42,shuffle=True)
##
sequences_train,sequences_validation = train_test_split(sequences_train,test_size=0.1111,stratify=sequences_train['label'],random_state=42,shuffle=True)
#####
print('Train: ',sequences_train['label'].value_counts(normalize=True)*100)
print('Validation: ',sequences_validation['label'].value_counts(normalize=True)*100)
print('Test: ',sequences_test['label'].value_counts(normalize=True)*100)
####Number of distinct labels in dataset#####
no_of_labels = int(sequences['label'].nunique())
###Calculate class weights with the inverse class frequency(inverse of each class percentage in the train dataset)
weights = sequences_train['label'].value_counts(normalize=True)
weights = torch.tensor([1/weights.loc[x] for x in sorted(list(weights.index))])
print(weights)
######Dataframe to HuggingFace Dataset#####

sequences_train = Dataset.from_pandas(sequences_train)
sequences_test = Dataset.from_pandas(sequences_test)
sequences_validation = Dataset.from_pandas(sequences_validation)
#input('WAIT')
###
#########Tokenizer#########
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # tokenizer
#tokenizer.padding_side = "left"
#
def tokenize_logs(entry):
  tokens = tokenizer(entry['text'],padding='max_length',truncation=True)
  tokens['labels'] = entry['label']
  return tokens
###TOKENIZING SUB DATASETS######

sequences_train = sequences_train.map(tokenize_logs,batched=True)
sequences_test = sequences_test.map(tokenize_logs,batched=True)
sequences_validation = sequences_validation.map(tokenize_logs,batched=True)
#####Metrics####
accuracy = evaluate.load("accuracy")
other_metrics = evaluate.combine(["precision","recall","f1"])
##
def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=-1)
  other_metrics_scores = other_metrics.compute(predictions=predictions,references=labels,average=None) #kathe klash
  accuracy_score = accuracy.compute(predictions=predictions,references=labels)["accuracy"] #epistrefei dict kai perno thn timh tou accuracy:
  all_metrics = {"accuracy":accuracy_score} #initialization
  for metric in other_metrics_scores: #appending
    for entry_pos,entry in enumerate(other_metrics_scores[metric]):
      all_metrics[f"{metric}_class_{entry_pos}"] = entry
  return all_metrics

####

####lora config

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=7,
    lora_alpha=14,
    lora_dropout=0.1,
   #MONO STO BERT target_modules=["query", "key", "value"]
   # target_modules=["query", "key","value"] #GIA TO DISTILBERT
   target_modules = ["q_lin","v_lin","k_lin"]
)

##Analoga poio einai to label einai to attack type h to label ????
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=no_of_labels) #modelo me ena extra layer gia to classification
#input('WAIT')
#Encapsulate
lora = get_peft_model(model,lora_config)
#orismata(pretrained modelo,configuration object)
###Feed model to CUDA
lora = lora.to(device)
###
training_arguments = TrainingArguments(
    output_dir = '/storage/data2/up1072604/run', #pou tha apothikeytei to modelo kai ta checkpoints. gia persistency to vazo sto /content/MyDrive/...
    overwrite_output_dir=True,  # MONO GIA THN PROTH FORA META PREPEI FALSE
    eval_strategy = "epoch", #to evaluation ginetai sto telos kathe synolou steps
    learning_rate=2e-5, #rythmos mathisis genika na einai mikros xoris akrotites. mikrh timh-> kalyterh genikeysh
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=2, #genika de thelei oute poly liga alla oute polla.poses fores to modelo tha iterarei to training dataset-koinos se posa epochs tha ginei to training
    weight_decay=0.01, #meiosh tou overfitting
    save_strategy= "no"
    #sose to state tou modelou sto telos tou epoch
    # Keep only the last 2 checkpoints to save disk space
    #kapoia akoma orismata + fakelos
)
####FOCAL LOSS FUNCTION###
#weights = torch.tensor([1.42,6.49,17.79,20.96,24.57,66666])

class SparseCategoricalFocalLoss(nn.Module):
  def __init__(self,gamma=2,alpha=None,reduction='mean'):
    super().__init__() #kalese ton constructor/init method ths iperkalsis
    self.gamma = gamma
    self.reduction = reduction
    self.alpha = alpha
  def forward(self,logits,labels):
    self.alpha = self.alpha.to(device)
    propabilities = F.softmax(logits,dim=-1) #pithanotites apo logits se probs meso softmax
    #diastaseis (batch,arithmos klaseon/output neurons)-(batch,2)
    ##fere ta labels se morfh (batch,1)
    labels = labels.view(-1,1)
    #ferta sth morfi (batch,)
    true_propabilities = propabilities.gather(1, labels).squeeze(1)
    #
    alpha_factor = self.alpha.gather(0,labels.view(-1))
    #
    loss = -alpha_factor * ((1-true_propabilities)**self.gamma) * torch.log(true_propabilities + 1e-8)
    #
    return loss.mean() if self.reduction == 'mean' else loss.sum()


#TRAINERS##
class ImbalancedTrainer(Trainer):
	def __init__(self,*args,loss_fn=None,**kwargs):
		super().__init__(*args,**kwargs)
		self.loss_fn = SparseCategoricalFocalLoss(gamma=2,alpha=weights,reduction='mean')
	def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
		#perno ta labels kai thn eksodo vasei ths eisodou sto modelo
		labels = inputs.pop('labels') #ground truth
		outputs = model(**inputs)
		logits = outputs.get('logits')
		#compute loss diafora logits kai anamenomenhs eksodou
		loss = self.loss_fn(logits,labels)
		#
		return (loss,outputs) if return_outputs else loss
####Instantiate ImbalancedTrainer#####

######
trainer = ImbalancedTrainer(
    model=lora, #modelo pou tha kanei to classification,
    args=training_arguments, #ta orismata tis ekpaideyshs apo pano
    #tokenizer=tokenizer#o tokenizer, an exo kanei hdh preprocessing-tokenizing LOGIKA den to xreiazomai
    train_dataset=sequences_train, #to synolo ekpaideyshs gia to fine-tuning
    eval_dataset=sequences_validation, #to synolo elegxou-aksiologisis gia to fine tuning-xoris ayto de tha aksiologithei h apodosi tou modelou mono tha ekpaideytei
    #tokenizer=tokenizer
    compute_metrics=compute_metrics #exei os apotelesma sto training tou modelou ektos apo to training/validation loss na emfanizontai kai oi times ton metrikon pou orisa sth synartisi kai epistrefontai apo ayth
    #
   )
###
##############CallBack for Training metrics########
class TrainingMetrics(TrainerCallback):
        def on_epoch_end(self,args,state,control,**kwargs):
                results = trainer.evaluate(eval_dataset=sequences_train)
                print('Training metrics on this epoch:')
                print(results)
####
#trainer.add_callback(TrainingMetrics)
##EXPLAINER(MODEL,TOKENIZER)
explainer = SequenceClassificationExplainer(lora,tokenizer)
explanations = []
def explain_flow(entry):
  global explanations
  attributions = explainer(entry['text'])
  html = explainer.visualize()
  explanations.append(f"{html.data}\n")
##########TRAIN && EVALUATE#########
trainer.train() # to orisma RESUME benei AFOU ginei interrupt!-to resume-from-checkpoint=directory
results = trainer.evaluate(eval_dataset=sequences_test) #ti apodosi fernei to modelo sto test dataset-orisma to test dataset
print(results)
##
######Apply the explanation function on each row#########

sequences_test.map(explain_flow)
######Auto closing and handling of file###############
with open('/storage/data2/up1072604/saves/explanations.html','w') as file:
  file.writelines(explanations)
