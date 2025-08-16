from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab import drive
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
drive.mount('/content/drive')
###ENVIRONMENT AND CONFIGURATIONS####
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))  # Show GPU details

PROBLEM = 'multi'

####################################

def Stats(dataset):
  dataset.info()
  print('Label distribution(number of classes):',dataset['label'].value_counts())
  print('Label distribution(percentage of classes):',dataset['label'].value_counts(normalize=True)*100)
##Loading data and printing information#######

data = pd.read_csv('/content/drive/MyDrive/Saves/cleaned.csv')
###Encoding labels####
if PROBLEM == 'multi':
  data['label'] = data['label'].map({'Benign':0,'DDoS':1,'Okiru':2,'PartOfAHorizontalPortScan':3,'C&C':4,'C&C-HeartBeat':4,'C&C-FileDownload':4,'C&C-Torii':4,'C&C-Mirai':4,'C&C-HeartBeat-FileDownload':4})
  data.dropna(axis=0,how='any',inplace=True) #If some categories are not included in the mapping they are removed
else:
  data['label'] = data['label'].apply(lambda x: 0 if x == 'Benign' else 1)
data.info()
'''
######Binary or Multi Classification for Edge-IIoTset####
if PROBLEM == 'binary':
  data.rename(columns={'Attack_label':'label'},inplace=True)
  data.drop(columns=['Attack_type'],inplace=True)
else:
  data.rename(columns={'Attack_type':'label'},inplace=True)
  data['label'] = data['label'].map({'Normal':0,'DDoS_UDP':1,'DDoS_ICMP':1,'DDoS_TCP':1,'DDoS_HTTP':1,'SQL_injection':2,'Uploading':2,'XSS':2,'Password':3,'Ransomware':3,'Backdoor':3,'Vulnerability_scanner':4,'Fingerprinting':4,'Port_Scanning':4})
  data.dropna(axis=0, how='any', inplace=True)
  data.drop(columns=['Attack_label'],inplace=True)
'''

Stats(data)
#
#########Drop duplicates###########
data = data.drop_duplicates(keep='first')
Stats(data)
#
conflicting_labels = data.groupby(data.columns.difference(['label']).tolist())['label'].nunique()

# Print number of cases where the same features have different labels
print("Rows with same features but different labels:", (conflicting_labels > 1).sum())
##############SHUFFLE####################
data = data.sample(frac=1,random_state=42) #shuffling
categorical_columns = data.drop(columns=['label']).select_dtypes(include=['object']).columns
numerical_columns = data.drop(columns=['label']).select_dtypes(exclude=['object']).columns

def evaluate(dataset):
  predictions = model.predict(dataset.drop(columns=['label']))
  true_labels = dataset['label']
  accuracy = accuracy_score(true_labels,predictions)
  metrics = precision_recall_fscore_support(true_labels,predictions,average=None)
  text = f"Accuracy: {accuracy:.6f}"
  dict_metrics = {0:'precision',1:'recall',2:'f1'}
  for index,metric in enumerate(metrics):
    if index == 3:
      break
    for pos,entry in enumerate(metric):
      text += f",{dict_metrics[index]}_class_{pos}:{entry:.6f}"
  print(text)
  matrix = confusion_matrix(true_labels,predictions) #y_true,y_pred
  figure = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=dataset['label'].unique())
  return figure

#
def Inspect(big,small,event):
  print(f'Distribution of train labels after {event}',big['label'].value_counts())
  print(f'Distribution of test labels after {event}',small['label'].value_counts())
  print("Train duplicates:", big.duplicated().sum())
  print("Test duplicates:", small.duplicated().sum())
  print(f"Common data between train & test: {len(big.merge(small,how='inner',on=big.columns.tolist()))}")
#

##########ONE HOT ENCODING#############

data = pd.get_dummies(data)
Stats(data)
###########SPLITTING############
data_train,data_test = train_test_split(data,test_size=0.2,stratify=data['label'],random_state=42,shuffle=True)
#Check
Inspect(data_train,data_test,'Removal of duplicates')

########SCALING###########
scaler = MinMaxScaler()
data_train[numerical_columns] = scaler.fit_transform(data_train[numerical_columns])
data_test[numerical_columns] = scaler.transform(data_test[numerical_columns])
print(data_train['label'].value_counts())
###Check###
Inspect(data_train,data_test,'Scaling')
#########MODEL#########
model = DecisionTreeClassifier()
model.fit(data_train.drop(columns=['label']),data_train['label'])
#####EVALUATION######
print('Eval stats:')
fig = evaluate(data_test)
fig.plot()
plt.show()
print('Training stats:')
fig = evaluate(data_train)
fig.plot()
plt.show()

