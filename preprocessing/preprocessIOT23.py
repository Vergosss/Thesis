import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount('/content/drive')
##########################
problem = 'MULTI'
##########READ DATA AVOIDING UNUSED COLUMN####
data = pd.read_csv('/content/drive/MyDrive/Data/iot23_combined_new.csv',index_col=0)
print('CHECK:',data.index.duplicated().any())
########Print class distribution and Dataframe dimensions ##########
print('Rows and columns of the dataset:',data.shape)
print('Distribution of labels:',data['label'].value_counts())
print('Distribution of labels:',data['label'].value_counts(normalize=True)*100)
print('Duration counts:',data['duration'].value_counts(dropna=False))

###########
####Columns before########
print('Columns of dataset:',data.columns)
drop_columns = ['ts','uid','id.orig_h','id.resp_h','local_orig','local_resp','orig_ip_bytes','resp_ip_bytes','history']
###Drop non useful features###
data.drop(drop_columns,axis=1,inplace=True)
print('After dropping columns:',data.columns)
###Null values per column###
print('Null values:',data.isnull().sum())
#
#####Cleaning columns with mismatched values###
data['orig_bytes'] = pd.to_numeric(data['orig_bytes'],errors='coerce') 
data['resp_bytes'] = pd.to_numeric(data['resp_bytes'],errors='coerce')
data['duration'] = pd.to_numeric(data['duration'],errors='coerce')

data.info()
###Null values per column###
print('Null values:',data.isnull().sum())
#### Replace null values with median value or mean or anything
data['orig_bytes'] = data['orig_bytes'].fillna(data['orig_bytes'].median()) #Median is 0 regardless of filling before or after splitting
data['resp_bytes'] = data['resp_bytes'].fillna(data['resp_bytes'].median()) #0 is a valid value and the median even after splitting
data['duration'] = data['duration'].fillna(data['duration'].median()) #median better for skewed distributions
print('Null values again: ',data.isnull().sum())
##

categorical_columns = data.select_dtypes(include=['object']).columns
numerical_columns = data.select_dtypes(exclude=['object']).columns
print(categorical_columns)
for column in numerical_columns:
    print(f"Counts of {column}:",data[column].value_counts(dropna=False))
for column in categorical_columns:
      print(f"Counts of {column}:",data[column].value_counts(dropna=False))

numeric_data = data[numerical_columns]

##########CORRELATION MATRIX################
data_matrix = numeric_data.corr()
#############Graph heatmap###########
plt.figure(figsize=(12, 8))
sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.show()
###########SAVE TO CSV###############
#data.to_csv('/content/drive/MyDrive/Saves/cleaned.csv',index=False,mode='w')
if problem == 'MULTI':
  data['label'] = data['label'].map({'Benign':0,'DDoS':1,'Okiru':2,'PartOfAHorizontalPortScan':3,'C&C':4,'C&C-HeartBeat':4,'C&C-FileDownload':4,'C&C-Torii':4,'C&C-Mirai':4,'C&C-HeartBeat-FileDownload':4})
  data.dropna(axis=0,how='any',inplace=True) #If some categories are not included in the mapping they are removed
else:
  data['label'] = data['label'].apply(lambda x: 0 if x == 'Benign' else 1)
###DROP DUPLICATES###
data = data.drop_duplicates(keep='first')
print('CHECK:',data.index.duplicated().any())
print('HOW:',data.index.duplicated().sum())
##############CONTEXT DICTIONARIES ########################
conn_state_dict = {'S0':'the originator tried to connect but the responder did not reply','S1':'the connection was established','S2':'the connection was established, the originator tried to close it but the responder did not reply','S3':'the connection was established, the responder tried to close it but the originator did not reply','SF':'the connection was established and terminated','SH':'the originator sent a SYN then a FIN but the responder did not send a SYN-ACK','SHR':'the responder sent a SYN-ACK then a FIN but the originator did not send a SYN','OTH':'no SYN observed','REJ':'the connection was rejected','RSTO':'the originator aborted the connection','RSTR':'the responder aborted the connection','RSTOS0':'the originator sent a SYN then a RST but the responder did not send a SYN-ACK','RSTRH':'the responder sent a SYN-ACK then a RST but the originator did not send a SYN'}
######CONVERT TO TEXT FOR TRANSFORMER/BERT-LIKE MODELS#############
sequences = []
def encode(entry):
    global sequences
    record = {}
    service = f"The application protocol utilized was {entry['service']}" if entry['service'] != '-' else f"The connection has no recorded application protocol"
    text = (
        f"This network flow shows a connection originating from port {entry['id.orig_p']} and destined to port {entry['id.resp_p']}. "
        f"{service}. "
        f"The connection lasted {entry['duration']} seconds. "
        f"The connection's transport layer protocol was {entry['proto']}. "
        f"The connection's state was set as {entry['conn_state']} meaning {conn_state_dict[entry['conn_state']]}. "
        f"The originator sent {entry['orig_bytes']} bytes of data, by sending {entry['orig_pkts']} packets. "
        f"The responder responded with {entry['resp_bytes']} bytes of data, by sending {entry['resp_pkts']} packets. "
        f"Packet loss was recorded as {entry['missed_bytes']} bytes.")
    record['text']  = text
    record['label'] = entry['label']
    sequences.append(record)
    return
#######APPLY ON EACH ROW AND CONVERT TO DATAFRAME AND SAVE#####
data.apply(encode, axis=1)
sequences = pd.DataFrame(sequences)
###
sequences.info()
print(sequences.shape)
print(sequences.head(1))

#sequences.to_csv('/content/drive/MyDrive/Saves/IoT23_sequences_binary.csv',index=False,mode='w')
sequences.to_csv('/content/drive/MyDrive/Saves/IoT23_sequences_multi.csv',index=False,mode='w')