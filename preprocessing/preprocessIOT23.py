import torch
import  pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')
##########################
problem = 'MULTI'
##########READ DATA AVOIDING UNUSED COLUMN####
data = pd.read_csv('/content/drive/MyDrive/Data/iot23_combined_new.csv',index_col=0)
########Print class distribution and Dataframe dimensions ##########
print('Rows and columns of the dataset:',data.shape)
print('Distribution of labels:',data['label'].value_counts())
print('Distribution of labels:',data['label'].value_counts(normalize=True)*100)
###########
####Dropping columns########

print('Columns of dataset:',data.columns)
drop_columns = ['ts','uid','id.orig_h','id.resp_h','local_orig','local_resp','duration','orig_ip_bytes','resp_ip_bytes']
##ektos ths label to timestamp kai to source/dest ip  de prosferoun pliroforia => epivarynsh modelou + overfitting apo to ip
#timestamps,ids epishs.
#mporo na droparo kai ypsila sisxetismena features
#ta local_orig,local_resp=> kenh '-' timh idia se oles tis eggrafes=> axristi stili de prosferei pliroforia
#orig_ip_bytes,resp_ip_bytes -> ipsila sisxetismena me ta orig_bytes,resp_bytes mporo na ta dioxo
#simiosi na do to resp_bytes me ta resp_pkts eno to idio de fainetai an isxyei me ta orig_bytes-orig_pkts
##
data.drop(drop_columns,axis=1,inplace=True)
print('After dropping columns:',data.columns)
##Null values
print('Null values:',data.isnull().sum())
#

#####Cleaning columns with mismatched values###
data['orig_bytes'] = pd.to_numeric(data['orig_bytes'],errors='coerce') #an ayto pou pao na kano noumero den einai arithmos to antikathista me NaN
data['resp_bytes'] = pd.to_numeric(data['resp_bytes'],errors='coerce')
data.info()
#Only column left is history
print('Null values again: ',data.isnull().sum())
#### Replace null values with median value or mean or anything
data['orig_bytes'] = data['orig_bytes'].fillna(data['orig_bytes'].median()) #median/mean/mode etc
data['resp_bytes'] = data['resp_bytes'].fillna(data['resp_bytes'].median())
print('Null values again: ',data.isnull().sum())
##
#einai kai ta orig_bytes kai ta resp_bytes alla exoun mismatched(datatype) epeidh einai object. Tha kano convert se numeric
#ta alla columns den exoun oute nulls oute mismatched type einai mia xara
numeric_columns = ['id.orig_p','id.resp_p','missed_bytes','orig_pkts','resp_pkts','orig_bytes','resp_bytes']

for column in numeric_columns:
    print(f"Counts of {column}:",data[column].value_counts(dropna=False))

#input('WAIT')
numeric_data = data[numeric_columns]
#numeric_data = data.select_dtypes(include=[np.number])
print('Numeric columns:',numeric_data.columns)

##########CORRELATION MATRIX################
data_matrix = numeric_data.corr()
#############Graph heatmap###########
plt.figure(figsize=(12, 8))
sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.show()
###########SAVE TO CSV###############
#data.to_csv('/content/drive/MyDrive/Saves/cleaned.csv',index=False,mode='w')

if problem = 'MULTI':
  data['label'] = data['label'].map({})
  data.dropna(axis=0,how='any',inplace=True)
else:
  data['label'] = data['label'].apply(lambda x: 0 if x == 'Benign' else 1)

data = data.drop_duplicates(keep='first')
data.info()
#input('WAIT')
##
######CONVERT TO TEXT FOR TRANSFORMER/BERT-LIKE MODELS#############
sequences = []
def encode(entry):
    #access entry column data with entry[column]
    global sequences
    record = {}
    service = f"The application protocol utilized was {entry['service']}" if entry['service'] != '-' else f"The connection has no recorded application protocol"
    text = f"The connection used the transport layer {entry['proto']} protocol, originating from port {entry['id.orig_p']} and destined to port {entry['id.resp_p']}. The connection's state was recorded as {entry['conn_state']}. {entry['orig_bytes']} bytes were sent from the originator and the responder sent {entry['resp_bytes']} bytes. Packet loss was {entry['missed_bytes']} bytes. The originator sent {entry['orig_pkts']} number of packets and the responder responded with {entry['resp_pkts']} number of packets. {service}, while the connection's state history is {entry['history']}"
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
sequences.to_csv('/content/drive/MyDrive/Saves/sequences_distinct_binary.csv',index=False,mode='w')
#sequences.to_csv('/content/drive/MyDrive/Saves/sequences_distinct.csv',index=False,mode='w')
#sequences.to_csv('/content/drive/MyDrive/Saves/sequences.csv',index=False,mode='w')