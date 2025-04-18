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
##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
###########event vectors/blocks###########
event_traces = pd.read_csv('/home/up1072604/data/Event_traces.csv',usecols=['BlockId','Label','Features'])
event_traces['Label'] = event_traces['Label'].map({'Success':0,'Fail':1})
#######event template- message#########
log_templates = pd.read_csv('/home/up1072604/data/HDFS.log_templates.csv')

##Dictionary of EventIds-Event text#######
event_dictionary = dict(zip(log_templates['EventId'],log_templates['EventTemplate']))

print(event_dictionary)

#Information
event_traces.info()
print(event_traces.shape)
print(event_traces.describe())
#
#######Convert feature event vector to text ###### 
def features_to_strings(entry):
  return " ".join([event_dictionary.get(eventID) for eventID in entry['Features'].replace('[','').replace(']','').split(',')])

###
##
print('Class distribution: ',event_traces['Label'].value_counts())
#
#
#input('WAIT')
event_traces.drop(columns=['BlockId'],inplace=True) #drop the block id
'''
EXPERIMENTING WITH UNDERSAMPLING ON WORSE HARDWARE
#drop a portion of 0's labels as the data is too big and to lower the frequency of this class
#
#event_traces_0 = event_traces[event_traces['Label']==0]
#event_traces_1 = event_traces[event_traces['Label']==1]
#event_traces_0 = event_traces_0.sample(frac=0.05,random_state=42)#an valo 0.035-0.3 eimai poly konta sta #arithmo klaseon 1
#event_traces = pd.concat([event_traces_0,event_traces_1],ignore_index=True)
#event_traces['Features'] = event_traces.apply(features_to_strings,axis=1)

#
#print('Class distribution: ',event_traces['Label'].value_counts())
#input('WAIT')
'''

#########Train test dev split##########
event_traces_train,event_traces_test = train_test_split(event_traces,test_size=0.1,random_state=42,stratify=event_traces['Label'],shuffle=True)
event_traces_train,event_traces_validation = train_test_split(event_traces_train,test_size=0.1111,stratify=event_traces_train['Label'],random_state=42,shuffle=True)
#
print('Class distribution train: ',event_traces_train['Label'].value_counts())
print('Class distribution validation: ',event_traces_validation['Label'].value_counts())
print('Class distribution test: ',event_traces_test['Label'].value_counts())
#######Apply on each row of the dataset###
event_traces_train['Features'] = event_traces_train.apply(features_to_strings,axis=1)
event_traces_validation['Features'] = event_traces_validation.apply(features_to_strings,axis=1)
event_traces_test['Features'] = event_traces_test.apply(features_to_strings,axis=1)
print(event_traces_train.sample(1))
#
event_traces_train = Dataset.from_pandas(event_traces_train)
event_traces_test = Dataset.from_pandas(event_traces_test)
event_traces_validation = Dataset.from_pandas(event_traces_validation)
##
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # tokenizer
#orisma ena line tou dataset opote epilego to column me to keimeno pou thelo na tokenaro
def tokenize_logs(entry):
  tokens = tokenizer(entry['Features'],padding='max_length',truncation=True)
  tokens['labels'] = entry['Label']
  return tokens

###############Tokenizing##########
event_traces_train = event_traces_train.map(tokenize_logs,batched=True)
event_traces_test = event_traces_test.map(tokenize_logs,batched=True)
event_traces_validation = event_traces_validation.map(tokenize_logs,batched=True)
#
#########Fine tuning meso ton parakato parametron##########
###METRICS####
#accuracy = %(predicted=true)
accuracy = evaluate.load("accuracy")
roc_auc = evaluate.load("roc_auc")
other_metrics = evaluate.combine(["precision","recall","f1"])
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    prediction_scores = expit(predictions) #expit(logits)
    roc_auc_score = roc_auc.compute(references=labels,prediction_scores=prediction_scores[:,1]) #y_true,y_labels
    predictions = np.argmax(predictions, axis=-1)
    other_metrics_scores = other_metrics.compute(predictions=predictions,references=labels,average=None) # kathe klash
    accuracy_score = accuracy.compute(predictions=predictions,references=labels)["accuracy"] #epistrefei dict
    return {"accuracy":accuracy_score,"precision_class_0":other_metrics_scores["precision"][0],"precision_class_1":other_metrics_scores["precision"][1],"recall_class_0":other_metrics_scores["recall"][0],"recall_class_1":other_metrics_scores["recall"][1],"f1_class_0":other_metrics_scores["f1"][0],"f1_class_1":other_metrics_scores["f1"][1],"roc_auc":roc_auc_score["roc_auc"]}


###MODEL###
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2).to(device) #modelo me ena extra layer gia to classification
###
training_arguments = TrainingArguments(
    output_dir = "/storage/data2/up1072604/run", #pou tha apothikeytei to modelo
    overwrite_output_dir = True,
    eval_strategy = "epoch", #to evaluation ginetai sto telos kathe epoch
    learning_rate=2e-5, #rythmos mathisis genika na einai mikros xoris akrotites. mikrh timh-> kalyterh genikeysh
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3, #genika de thelei oute poly liga alla oute polla.poses fores to modelo tha iterarei to training dataset-koinos se posa epochs tha ginei to training
    weight_decay=0.01 #meiosh tou overfitting
    #kapoia akoma orismata + fakelos
)

#########
trainer = Trainer(
    model=model, #modelo pou tha kanei to classification,
    args=training_arguments, #ta orismata tis ekpaideyshs apo pano
    #tokenizer=tokenizer#o tokenizer, an exo kanei hdh preprocessing-tokenizing LOGIKA den to xreiazomai
    train_dataset= event_traces_train, #to synolo ekpaideyshs gia to fine-tuning
    eval_dataset=event_traces_validation, #to synolo elegxou-aksiologisis gia to fine tuning-xoris ayto de tha aksiologithei h apodosi tou modelou mono tha ekpaideytei
    #tokenizer=tokenizer
    compute_metrics=compute_metrics #exei os apotelesma sto training tou modelou ektos apo to training/validation loss na emfanizontai kai oi times ton metrikon pou orisa sth synartisi kai epistrefontai apo ayth
    #
    )
### Train the model
trainer.train()
results = trainer.evaluate(eval_dataset=event_traces_test) #ti apodosi fernei to modelo sto test dataset
print(results)

