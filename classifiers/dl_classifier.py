!pip install focal_loss
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from google.colab import drive
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from focal_loss import SparseCategoricalFocalLoss
drive.mount('/content/drive')
#####ENVIRONMENT AND CONFIGURATIONS#######
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))  # Show GPU details
#print(tf.__version__)

PROBLEM = 'multi'
####
def Stats(dataset):
  dataset.info()
  print('Label distribution(number of classes):',dataset['label'].value_counts())
  print('Label distribution(percentage of classes):',dataset['label'].value_counts(normalize=True)*100)
######Loading data and printing information#######

data = pd.read_csv('/content/drive/MyDrive/Saves/cleaned.csv')

###Encoding labels####
if PROBLEM == 'multi':
  data['label'] = data['label'].map({'Benign':0,'DDoS':1,'Okiru':2,'PartOfAHorizontalPortScan':3,'C&C':4,'C&C-HeartBeat':4,'C&C-FileDownload':4,'C&C-Torii':4,'C&C-Mirai':4,'C&C-HeartBeat-FileDownload':4})
  data.dropna(axis=0,how='any',inplace=True) #If some categories are not included in the mapping they are removed
else:
  data['label'] = data['label'].apply(lambda x: 0 if x == 'Benign' else 1)
data.info()

######Binary or Multi Classification####
'''
For Edge-IIoTset
if PROBLEM == 'binary':
  data.rename(columns={'Attack_label':'label'},inplace=True)
  data.drop(columns=['Attack_type'],inplace=True)
else:
  data.rename(columns={'Attack_type':'label'},inplace=True)
  data['label'] = data['label'].map({'Normal':0,'DDoS_UDP':1,'DDoS_ICMP':1,'DDoS_TCP':1,'DDoS_HTTP':1,'SQL_injection':2,'Uploading':2,'XSS':2,'Password':3,'Ransomware':3,'Backdoor':3,'Vulnerability_scanner':4,'Fingerprinting':4,'Port_Scanning':4})
  data.dropna(axis=0, how='any', inplace=True)
  data.drop(columns=['Attack_label'],inplace=True)
'''

#
###################Drop duplicates##########
data = data.drop_duplicates(keep='first')
Stats(data)

############SHUFFLE############
data = data.sample(frac=1,random_state=42)
Stats(data)
###
categorical_columns = data.drop(columns=['label']).select_dtypes(include=['object']).columns
numerical_columns = data.drop(columns=['label']).select_dtypes(exclude=['object']).columns

data = pd.get_dummies(data)
Stats(data)
##Spliting train validation test
data_train,data_test = train_test_split(data,test_size=0.1,stratify=data['label'],random_state=42,shuffle=True)
data_train,data_validation = train_test_split(data_train,test_size=0.1111,stratify=data_train['label'],random_state=42,shuffle=True)
###Scaling###
scaler = MinMaxScaler()
scaler.fit(data_train[numerical_columns])
data_train[numerical_columns] = scaler.transform(data_train[numerical_columns])
data_validation[numerical_columns] = scaler.transform(data_validation[numerical_columns])
data_test[numerical_columns] = scaler.transform(data_test[numerical_columns])
dim = data_train.shape[1] -1
print(dim)
#
###Architecture###
model = Sequential()

model.add(Dense(units=64, input_dim=dim, activation='relu')) #First Hidden layer
model.add(Dropout(0.3)) #dropout gia to overfitting
 #
model.add(Dense(units=32, activation='relu')) #Second hidden layer
model.add(Dropout(0.3))

model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.3))
#model.add(Dense(units=4, activation='relu')) #Second hidden layer

model.add(Dense(units=5,activation='softmax'))
#############EVALUATION FUNCTION INCLUDING SIGMOID ACTIVATION FOR BINARY CLASSIFICATION, SOFTMAX ACTIVATION FOR ANY NUMBER OF CLASSES AND EVALUATION PER EPOCH OR AT THE END
def evaluate(dataset):
  test_loss, test_accuracy = model.evaluate(dataset.drop(columns=['label']),dataset['label'])
  propabilities = model.predict(dataset.drop(columns=['label']))
  predictions = np.argmax(propabilities,axis=1)
  true_labels = dataset['label']
  accuracy = accuracy_score(true_labels,predictions)
  metrics = precision_recall_fscore_support(true_labels,predictions,average=None)
  text = f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.6f}, Accuracy:{accuracy:.6f}"
  dict_metrics = {0:'precision',1:'recall',2:'f1'}
  for index,metric in enumerate(metrics):
    if index == 3:
      break
    for pos,entry in enumerate(metric):
      text += f",{dict_metrics[index]}_class_{pos}:{entry:.6f}"
  print(text)
####

#####
class GeneralMetric(tf.keras.callbacks.Callback):
  def __init__(self,validation_data):
    super().__init__()
    self.validation_data = validation_data
  def on_epoch_end(self,epoch,logs=None):
    eval_loss, eval_accuracy = self.model.evaluate(self.validation_data[0],self.validation_data[1],verbose=0)
    propabilities = self.model.predict(self.validation_data[0])
    predictions = np.argmax(propabilities,axis=1) #class with biggest propabilities
    #ground truth
    true_labels = self.validation_data[1]
    accuracy = accuracy_score(true_labels,predictions)
    metrics = precision_recall_fscore_support(true_labels,predictions,average=None)
    text = f"Validation loss: {eval_loss:.4f}, Validation accuracy: {eval_accuracy:.6f},Accuracy:{accuracy:.6f}"
    dict_metrics = {0:'precision',1:'recall',2:'f1'}
    for index,metric in enumerate(metrics):
      if index == 3:
        break
      for pos,entry in enumerate(metric):
        text += f",{dict_metrics[index]}_class_{pos}:{entry:.6f}"
    print(text)

##Class weights###
weights = data_train['label'].value_counts(normalize=True)
weights = torch.tensor([1/weights.loc[x] for x in sorted(list(weights.index))])
print(weights)

######
###### Compile the model with optimizer,learning rate and loss function #####
model.compile(optimizer=Adam(learning_rate=2e-5), loss=SparseCategoricalFocalLoss(gamma=2,class_weight=weights), metrics=['accuracy'])
####SUMMARIZING INFO####
model.summary()
#Initialize the objects
general_metric = GeneralMetric(validation_data=(data_validation.drop(columns=['label']),data_validation['label']))
#
########Train the model########
model.fit(data_train.drop(columns=['label']),data_train['label'],epochs=20,batch_size=16,validation_data=(data_validation.drop(columns=['label']),data_validation['label']),callbacks=[general_metric])
#
print('Evaluating on the test set:')
#### Evaluation on test set #######
evaluate(data_test)
##### Evaluation on the training set to check for overfitting ####
print('Evaluating on the training set:')
evaluate(data_train)