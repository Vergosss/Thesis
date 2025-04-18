!pip install focal_loss
import  pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from google.colab import drive
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from focal_loss import SparseCategoricalFocalLoss
drive.mount('/content/drive')
#####ENVIRONMENT AND CONFIGURATIONS#######
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))  # Show GPU details
#print(tf.__version__)
print(pd.__version__)
input('WAIT')
PROBLEM = 'multiclass'
####
def Stats(dataset):
  dataset.info()
  print('Label distribution(number of classes):',dataset['label'].value_counts())
  print('Label distribution(percentage of classes):',dataset['label'].value_counts(normalize=True)*100)
#
######Loading data and printing information#######

data = pd.read_csv('/content/drive/MyDrive/Saves/EdgeIIoT.csv')
#
###Encoding labels####
#data['label'] = data['label'].apply(lambda x: 0 if x == 'Benign' else 1)
#data['label'] = data['label'].map({'Normal':0,'DDoS_UDP':1,'DDoS_ICMP':1,'DDoS_TCP':1,'DDoS_HTTP':1,'SQL_injection':2,'Uploading':2,'XSS':2,'Password':3,'Ransomware':3,'Backdoor':3,'Vulnerability_scanner':4,'Fingerprinting':4,'Port_Scanning':4,'MITM':5})
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

###########Drop duplicates############
data = data.drop_duplicates(keep='first')
Stats(data)

#
###################Drop duplicates##########
data = data.drop_duplicates(keep='first')
Stats(data)

############SHUFFLE############
data = data.sample(frac=1,random_state=42)
Stats(data)
###
##One hot encoding
categorical_columns = data.drop(columns=['label']).select_dtypes(include=['object']).columns
###Afairo h to label h to type analoga to task kai exo mono label opote eimai safe
numerical_columns = data.drop(columns=['label']).select_dtypes(exclude=['object']).columns

#features = numerical_columns + categorical_columns
#
data = pd.get_dummies(data) #prin to splitting giati borei kapoies katigories na min yparxoun sto validation/test
Stats(data)
#opote tha eixa ligoteres diastaseis sto validation/test
##Spliting train validation test
data_train,data_test = train_test_split(data,test_size=0.1,stratify=data['label'],random_state=42,shuffle=True)
data_train,data_validation = train_test_split(data_train,test_size=0.1111,stratify=data_train['label'],random_state=42,shuffle=True)

scaler = MinMaxScaler()
scaler.fit(data_train[numerical_columns]) #vasei tou training set kane scale ta alla
data_train[numerical_columns] = scaler.transform(data_train[numerical_columns])
data_validation[numerical_columns] = scaler.transform(data_validation[numerical_columns])
data_test[numerical_columns] = scaler.transform(data_test[numerical_columns])
#
dim = data_train.shape[1] -1
print(dim)
#
##Architecture##
model = Sequential()

model.add(Dense(units=6, input_dim=dim, activation='relu')) #gelu activation kai weight decay
#model.add(Dropout(0.3)) #dropout gia to overfitting
 #
model.add(Dense(units=5, activation='relu')) #32 neyrones, gelu activation kai weight decay parametro
#model.add(Dropout(0.3))
 #
#model.add(Dense(units=16, activation='relu')) #16 neyrones, gelu activation kai weight decay parametro

#
model.add(Dense(units=6,activation='softmax'))
#
#
#############EVALUATION FUNCTION INCLUDING SIGMOID ACTIVATION FOR BINARY CLASSIFICATION, SOFTMAX ACTIVATION FOR ANY NUMBER OF CLASSES AND EVALUATION PER EPOCH OR AT THE END
def evaluate(dataset):
  test_loss, test_accuracy = model.evaluate(dataset.drop(columns=['label']),dataset['label'])
  propabilities = model.predict(dataset.drop(columns=['label']))
  predictions = np.argmax(propabilities,axis=1)
  true_labels = dataset['label']
  accuracy = accuracy_score(true_labels,predictions)
  metrics = precision_recall_fscore_support(true_labels,predictions,average=None)
  text = f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}"
  dict_metrics = {0:'precision',1:'recall',2:'f1'}
  for index,metric in enumerate(metrics):
    if index == 3:
      break
    for pos,entry in enumerate(metric):
      text += f",{dict_metrics[index]}_class_{pos}:{entry:.4f}"
  print(text)
####
class EvaluatePerEpochMetrics(tf.keras.callbacks.Callback):
  def __init__(self,validation_data):
    super().__init__()
    self.validation_data = validation_data
  def on_epoch_end(self, epoch, logs=None):
    #to validation data einai tuple typou (x_val,y_val)
    eval_loss, eval_accuracy = self.model.evaluate(self.validation_data[0], self.validation_data[1], verbose=0)
    #gia ta validation data provlepse vgale pithanotites
    propabilities = self.model.predict(self.validation_data[0])
    # Pithanotites se labels metafrash
    predictions= (propabilities > 0.5).astype("int32")
	  #Pare to ground truth diladi ta labels
    true_labels = self.validation_data[1]
    accuracy = accuracy_score(true_labels,predictions)
    metrics = precision_recall_fscore_support(true_labels,predictions,average=None)
    print(f'Validation loss: {eval_loss:.4f}, Validation accuracy:{eval_accuracy:.4f},precision_class_0:{metrics[0][0]:.4f},precision_class_1:{metrics[0][1]:.4f},recall_class_0:{metrics[1][0]:.4f},recall_class_1:{metrics[1][1]:.4f},f1_class_0:{metrics[2][0]:.4f},f1_class_1:{metrics[2][1]:.4f}')
#####
class GeneralMetric(tf.keras.callbacks.Callback):
  def __init__(self,validation_data):
    super().__init__()
    self.validation_data = validation_data
  def on_epoch_end(self,epoch,logs=None):
    eval_loss, eval_accuracy = self.model.evaluate(self.validation_data[0],self.validation_data[1],verbose=0)
    propabilities = self.model.predict(self.validation_data[0])
    #propabilities h logits?
    predictions = np.argmax(propabilities,axis=1) #perno thn klash me th megaliterh pithanotita
    #ground truth
    true_labels = self.validation_data[1]
    accuracy = accuracy_score(true_labels,predictions)
    metrics = precision_recall_fscore_support(true_labels,predictions,average=None)
    text = f"Validation loss: {eval_loss:.4f}, Validation accuracy: {eval_accuracy:.4f}"
    dict_metrics = {0:'precision',1:'recall',2:'f1'}
    for index,metric in enumerate(metrics):
      if index == 3:
        break
      for pos,entry in enumerate(metric):
        text += f",{dict_metrics[index]}_class_{pos}:{entry:.4f}"
    print(text)

##Class weights###
weight_0 = data_train.shape[0]/(2*data_train['label'].value_counts()[0])
weight_1 = data_train.shape[0]/(2*data_train['label'].value_counts()[1])
class_weights = {0:weight_0,1:weight_1}
print(class_weights)

######
###### Compile the model with optimizer,learning rate and loss function #####
model.compile(optimizer=Adam(learning_rate=2e-5), loss=SparseCategoricalFocalLoss(gamma=2), metrics=['accuracy'])
####SUMMARIZING INFO####
model.summary()
#Initialize the objects
eval_func = EvaluatePerEpochMetrics(validation_data=(data_validation.drop(columns=['label']),data_validation['label']))
general_metric = GeneralMetric(validation_data=(data_validation.drop(columns=['label']),data_validation['label']))
#
########Train the model########
#
model.fit(data_train.drop(columns=['label']),data_train['label'],epochs=20,batch_size=16,validation_data=(data_validation.drop(columns=['label']),data_validation['label']),callbacks=[general_metric])
#
print('Evaluating on the test set:')
#### Evaluation on test set #######
evaluate(data_test)
##### Evaluation on the training set to check for overfitting ####
print('Evaluating on the training set:')
evaluate(data_train)