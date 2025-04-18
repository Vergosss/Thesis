from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import  pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from google.colab import drive
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
drive.mount('/content/drive')
#ENVIRONMENT AND CONFIGURATIONS####
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))  # Show GPU details
print(matplotlib.__version__)
input('WAIT')
PROBLEM = 'multiclass'

####################################

def Stats(dataset):
  dataset.info()
  print('Label distribution(number of classes):',dataset['label'].value_counts())
  print('Label distribution(percentage of classes):',dataset['label'].value_counts(normalize=True)*100)
##Loading data and printing information#######

data = pd.read_csv('/content/drive/MyDrive/Saves/EdgeIIoT.csv')
###Encoding labels####
#data['label'] = data['label'].apply(lambda x: 0 if x == 'Benign' else 1)
#data['label'] = data['label'].map({'Normal':0,'DDoS_UDP':1,'DDoS_ICMP':1,'DDoS_TCP':1,'DDoS_HTTP':1,'SQL_injection':2,'Uploading':2,'XSS':2,'Password':3,'Ransomware':3,'Backdoor':3,'Vulnerability_scanner':4,'Fingerprinting':4,'Port_Scanning':4,'MITM':5})
data.info()
######Binary or Multi Classification####
if PROBLEM == 'binary':
  data.rename(columns={'Attack_label':'label'}, inplace=True)
  data.drop(columns=['Attack_type'],inplace=True)
else:
  data.rename(columns={'Attack_type':'label'},inplace=True)
  #data['label'] = data['label'].map({'Normal':0,'DDoS_UDP':1,'DDoS_ICMP':2,'SQL_injection':3,'Password':4,'Vulnerability_scanner':5})
  data['label'] = data['label'].map({'Normal':0,'DDoS_UDP':1,'DDoS_ICMP':1,'DDoS_TCP':1,'DDoS_HTTP':1,'SQL_injection':2,'Uploading':2,'XSS':2,'Password':3,'Ransomware':3,'Backdoor':3,'Vulnerability_scanner':4,'Fingerprinting':4,'Port_Scanning':4,'MITM':5})
  data.dropna(axis=0, how='any', inplace=True)
  data.drop(columns=['Attack_label'],inplace=True)

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
#
#
#numerical_columns = ['id.orig_p','id.resp_p','orig_bytes','resp_bytes','missed_bytes','orig_pkts','resp_pkts']
categorical_columns = data.drop(columns=['label']).select_dtypes(include=['object']).columns
numerical_columns = data.drop(columns=['label']).select_dtypes(exclude=['object']).columns

#features = numerical_columns + categorical_columns
#
def evaluate(dataset):
  predictions = model.predict(dataset.drop(columns=['label']))
  #y_pred = (propabilities > 0.5).astype('int32')
  true_labels = dataset['label']
  accuracy = accuracy_score(true_labels,predictions)
  metrics = precision_recall_fscore_support(true_labels,predictions,average=None)
  text = f"Accuracy: {accuracy:.4f}"
  dict_metrics = {0:'precision',1:'recall',2:'f1'}
  for index,metric in enumerate(metrics):
    if index == 3:
      break
    for pos,entry in enumerate(metric):
      text += f",{dict_metrics[index]}_class_{pos}:{entry:.4f}"
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

####If there are not any categorical(object,category) features in the dataset the rest of numerical type dont get affected at all.
data = pd.get_dummies(data)
Stats(data)
###########SPLITTING############
data_train,data_test = train_test_split(data,test_size=0.2,stratify=data['label'],random_state=42,shuffle=True)
#data_train = data_train.drop_duplicates(keep='first')
#
#Check
Inspect(data_train,data_test,'Removal of duplicates')

########SCALING###########
scaler = MinMaxScaler()
data_train[numerical_columns] = scaler.fit_transform(data_train[numerical_columns])
data_test[numerical_columns] = scaler.transform(data_test[numerical_columns])
#Check
Inspect(data_train,data_test,'Scaling')
#
#data_train = data_train.drop_duplicates(keep='first')
#Check
Inspect(data_train,data_test,'Removal of duplicates')

input('WAIT')
########################
'''
EXPERIMENTING WITH SMOTE FOR NOMINAL AND CATEGORICAL FEATURES(ALONGSIDE NUMERICAL FEATURES)
###########SPLITTING############
data_train,data_test = train_test_split(data,test_size=0.2,stratify=data['label'],random_state=42,shuffle=True)
###Random Oversampling SMOTE mono sto train dataset-PERNO TA INDICES TOU TRAIN SUBSET######

categorical_columns = data_train.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_indices = [data_train.columns.get_loc(col) for col in categorical_columns]  # Get indices in data_train
#
###########OVERSAMPLING-SMOTE##############
oversampler = SMOTENC(categorical_features=categorical_indices,sampling_strategy=0.4,random_state=42) #minority = tade % tis majority
X,y = oversampler.fit_resample(data_train.drop(columns=['label']),data_train['label'])
data_train = pd.concat([X,y],axis=1)#sygxoneyse ta ana stiles
#
Inspect(data_train,data_test,'SMOTE')
#Drop duplicates
data_train = data_train.drop_duplicates(keep='first')#oxi duplicates sto train set logw smote(poli spanio)-na pesoun idia me kapoia tou train hdh
#
##Check again
Inspect(data_train,data_test,'Removal of duplicates')

########SCALING###########
scaler = MinMaxScaler()
data_train[numerical_columns] = scaler.fit_transform(data_train[numerical_columns])
data_test[numerical_columns] = scaler.transform(data_test[numerical_columns])
#Check
Inspect(data_train,data_test,'Scaling')
#Drop duplicates
data_train = data_train.drop_duplicates(keep='first')
##Check again
Inspect(data_train,data_test,'Removal of duplicates')
#

#
input('WAIT')
###One Hot encoding For ML/DL Models###
data_train = pd.get_dummies(data_train)
data_test = pd.get_dummies(data_test)
###Align the datasets###
data_train, data_test = data_train.align(data_test,join='left',axis=1,fill_value=0)
#Kane align vasei ton stilon tou aristerou merous(tou training), kane align stis stiles kai gemise tis stiles pou den yparxoun me mhden
'''
#########MODEL#########
#to scale_pos_weight = pleiopsifiki klash/meiopsifiki klasi
#6.69
model = SGDClassifier()
model.fit(data_train.drop(columns=['label']),data_train['label'])
#####EVALUATION######
fig = evaluate(data_test)
fig.plot()
plt.show()
print('Training stats:')
fig = evaluate(data_train)
fig.plot()
plt.show()

