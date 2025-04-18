import re
import pickle
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from transformers import BertTokenizer,BertForSequenceClassification,Trainer,TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
from scipy.special import expit
#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#
windows = pd.read_csv('/home/up1072604/saves/windows.csv',index_col=0) #index_col=0 agnoei thn unamed sthlh

print(windows.columns)
print('Class distribution: ',windows['labels'].value_counts())
#
print('Null text: ',windows['text'].isnull().sum())
print('Null labels: ',windows['labels'].isnull().sum())
#
windows['text'] = windows['text'].fillna('')
print('Null text: ',windows['text'].isnull().sum())

#
#input('WAIT')
##
windows_train,windows_test = train_test_split(windows,test_size=0.1,random_state=42,stratify=windows['labels'],shuffle=True)
##
windows_train,windows_validation = train_test_split(windows_train,test_size=0.1111,random_state=42,stratify=windows_train['labels'],shuffle=True)
######Dataframe to HuggingFace Dataset#####

windows_train = Dataset.from_pandas(windows_train)
windows_test = Dataset.from_pandas(windows_test)
windows_validation = Dataset.from_pandas(windows_validation)
###########
###TOKENIZER && TOKENIZING####
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base") # tokenizer

def tokenize_logs(entry):
  tokens = tokenizer(entry['text'],padding='max_length',truncation=True)
  #tokens['labels'] = entry['Label']
  return tokens
###TOKENIZING SUB DATASETS######
windows_train = windows_train.map(tokenize_logs,batched=True)
windows_test = windows_test.map(tokenize_logs,batched=True)
windows_validation = windows_validation.map(tokenize_logs,batched=True)

##METRICS####
accuracy = evaluate.load("accuracy")
other_metrics = evaluate.combine(["precision","recall","f1"])
roc_auc = evaluate.load("roc_auc")
#combine all metrics to a dictionary
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    prediction_scores = expit(predictions) #expit(logits)
    roc_auc_score = roc_auc.compute(references=labels,prediction_scores=prediction_scores[:,1]) #y_true,y_score
    predictions = np.argmax(predictions, axis=-1)
    other_metrics_scores = other_metrics.compute(predictions=predictions,references=labels,average=None) #kathe klash
    accuracy_score = accuracy.compute(predictions=predictions,references=labels)["accuracy"] #epistrefei dict kai perno thn timh tou accuracy:
    return {"accuracy":accuracy_score,"precision_class_0":other_metrics_scores["precision"][0],"precision_class_1":other_metrics_scores["precision"][1],"recall_class_0":other_metrics_scores["recall"][0],"recall_class_1":other_metrics_scores["recall"][1],"f1_class_0":other_metrics_scores["f1"][0],"f1_class_1":other_metrics_scores["f1"][1],"roc_auc":roc_auc_score["roc_auc"]}

###MODEL###
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base",num_labels=2).to(device) #modelo me ena extra layer gia to classification
###TRAINING ARGS#####
training_arguments = TrainingArguments(
    output_dir = "/storage/data2/up1072604/run", #pou tha apothikeytei to modelo kai ta checkpoints. gia persistency to vazo sto /content/MyDrive/...
    overwrite_output_dir=True,  # MONO GIA THN PROTH FORA META PREPEI FALSE
    eval_strategy = "epoch", #to evaluation ginetai sto telos kathe synolou steps
    learning_rate=2e-5, #rythmos mathisis genika na einai mikros xoris akrotites. mikrh timh-> kalyterh genikeysh
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3, #genika de thelei oute poly liga alla oute polla.poses fores to modelo tha iterarei to training dataset-koinos se posa epochs tha ginei to training
    weight_decay=0.01, #meiosh tou overfitting
    save_strategy = "no"
    #kapoia akoma orismata + fakelos
)
####TRAINER###
trainer = Trainer(
    model=model, #modelo pou tha kanei to classification,
    args=training_arguments, #ta orismata tis ekpaideyshs apo pano
    #tokenizer=tokenizer#o tokenizer, an exo kanei hdh preprocessing-tokenizing LOGIKA den to xreiazomai
    train_dataset=windows_train, #to synolo ekpaideyshs gia to fine-tuning
    eval_dataset=windows_validation, #to synolo elegxou-aksiologisis gia to fine tuning-xoris ayto de tha aksiologithei h apodosi tou modelou mono tha ekpaideytei
    #tokenizer=tokenizer
    compute_metrics=compute_metrics #exei os apotelesma sto training tou modelou ektos apo to training/validation loss na emfanizontai kai oi times ton metrikon pou orisa sth synartisi kai epistrefontai apo ayth
    #
    )

####TRAIN && EVALUATE
trainer.train() #to orisma RESUME benei AFOU ginei interrupt!-to resume-from-checkpoint=directory
results = trainer.evaluate(eval_dataset=windows_test) #ti apodosi fernei to modelo sto test dataset-orisma to test dataset
#print(results)
##STORE RESULTS TO FILE##
with open('results_BGL_distilroberta.pkl','ab') as results_pkl:
    pickle.dump(results,results_pkl)
   

