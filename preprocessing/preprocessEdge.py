import torch
import  pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from google.colab import drive
from sys import exit
drive.mount('/content/drive')

####
problem = 'MULTICLASS'

data = pd.read_csv('/content/drive/MyDrive/Data/DNN-EdgeIIoT-dataset.csv')
#########CLASS DISTRIBUTION WHETHER BINARY OR MULTI ###########
print('Distribution of labels:',data['Attack_label'].value_counts(normalize=True)*100)
##
print('Distribution of types:',data['Attack_type'].value_counts(normalize=True)*100)


############DROP COLUMNS-IRRELEVANT FEATURES ##############
print('Columns of dataset BEFORE:',data.columns)
drop_columns = ['ip.src_host','ip.dst_host','frame.time','icmp.unused','icmp.transmit_timestamp','arp.src.proto_ipv4','arp.dst.proto_ipv4','mqtt.msg_decoded_as', 'http.file_data','http.request.full_uri','http.request.uri.query','tcp.options','http.tls_port','dns.qry.type','mqtt.msg','mqtt.protoname','mqtt.topic','mbtcp.trans_id','mbtcp.unit_id','mqtt.ver','mqtt.conack.flags','arp.opcode','arp.hw.size','mqtt.conflag.cleansess','dns.retransmit_request','dns.retransmit_request_in','tcp.checksum','icmp.seq_le','tcp.payload','tcp.flags.ack','dns.qry.name.len','udp.time_delta','udp.port','dns.qry.name','dns.qry.type','dns.qry.qu','dns.retransmission','mbtcp.len','mqtt.topic_len','mqtt.proto_len','http.referer','http.request.version','http.request.method']
data.drop(drop_columns,axis=1,inplace=True)
print(f'AFTER dropping columns: {data.columns} number: {len(data.columns)}')
##Counts of columns
print('Nulls : ',data.isnull().sum().sum())
print('Counts:',data["tcp.dstport"].value_counts())
print('Counts:',data["tcp.dstport"].nunique())

data.info()

##
mask = pd.to_numeric(data['tcp.srcport'], errors='coerce').isna()
print('Rows',data.loc[mask,'tcp.srcport'].value_counts())
##
######## NUMERIZATION AND DATA CLEANING ############
categorical_columns = data.select_dtypes(include=['object']).columns
numerical_columns = data.select_dtypes(exclude=['object']).columns
#
data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric,errors='coerce')
data['tcp.srcport'] = pd.to_numeric(data['tcp.srcport'],errors='coerce')


###
print('Nulls after numerization',data[numerical_columns].isna().sum())
data.info()
#REMOVE THE VERY FEW ROWS WITH NULL TCP PORT #####
data.dropna(axis=0, how='any', inplace=True)
###############DUPLICATE CHECKING###########
print('Num of duplicates in dataset:',data.duplicated().sum())
data.info()
print('Attack types: ',data['Attack_type'].value_counts())
###########CORRELATION MATRIX##################
data_matrix = data.corr(numeric_only=True)
plt.figure(figsize=(18, 14))
sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.savefig('/content/drive/MyDrive/Saves/Features.png')
plt.show()
######FEATURE - LABEL CORRELATION IN DESCENDING ORDER#####
print(data_matrix['Attack_label'].abs().sort_values(ascending=False))

##################################
########SAVING TO CSV ###########
#data.to_csv('/content/drive/MyDrive/Saves/EdgeIIoT.csv',index=False,mode='w')
#exit(0)
if problem == 'BINARY':
  data.rename(columns={'Attack_label':'label'}, inplace=True)
  data.drop(columns=['Attack_type'],inplace=True)
else:
  data.rename(columns={'Attack_type':'label'},inplace=True)
  #data['label'] = data['label'].map({'Normal':0,'DDoS_UDP':1,'DDoS_ICMP':2,'SQL_injection':3,'Password':4,'Vulnerability_scanner':5})
  data['label'] = data['label'].map({'Normal':0,'DDoS_UDP':1,'DDoS_ICMP':1,'DDoS_TCP':1,'DDoS_HTTP':1,'SQL_injection':2,'Uploading':2,'XSS':2,'Password':3,'Ransomware':3,'Backdoor':3,'Vulnerability_scanner':4,'Fingerprinting':4,'Port_Scanning':4,'MITM':5})
  data.dropna(axis=0, how='any', inplace=True)
  data.drop(columns=['Attack_label'],inplace=True)

data = data.drop_duplicates(keep='first') 
##############CONTEXT DICTIONARIES ########################
http_response_dict = {0.0:'No HTTP response was received',1.0:'An HTTP reponse was received'}
mqtt_conflags_dict = {0:'',2:''}
tcp_flags_dict ={0.0:'No TCP flags set',16.0:'ACK flag set',4.0:'RST flag set',2.0:'SYN flag set',24.0:'PSH and ACK flags set',17.0:'ACK and FIN flags set',18.0:'SYN and ACK flags set',25.0: 'ACK, PSH, and FIN flags set',20.0:'ACK and RST flags set'}
mqtt_msgtype_dict ={}
#########CONVERT TO TEXT FOR TRANSFORMER/BERT MODELS########
sequences = []
def encode(entry):
  global sequences
  record = {}
  #text = f"This packet shows a connection originating from port {entry['tcp.srcport']} and destined to port {entry['tcp.dstport']}. The length of the tcp packet was {entry['tcp.len']} and its sequence number {entry['tcp.seq']}. The tcp connection's FIN, RST, SYN, SYNACK flags were set to {entry['tcp.connection.fin']}, {entry['tcp.connection.rst']}, {entry['tcp.connection.syn']} and {entry['tcp.connection.synack']} respectively, resulting in flags value of {entry['tcp.flags']}. The size of the transferred HTTP data was {entry['http.content_length']} while the http response code was {entry['http.response']}. The MQTT message length was {entry['mqtt.len']} and the type of the message was {entry['mqtt.msgtype']}. The MQTT connection flags had a value of {entry['mqtt.conflags']} and the MQTT header flags were set to {entry['mqtt.hdrflags']}. The checksum value for the ICMP protocol was {entry['icmp.checksum']}. A UDP communication stream was identified with stream id of {entry['udp.stream']}. The packet's acknowledgement number was {entry['tcp.ack']} and its raw equivalent {entry['tcp.ack_raw']}"
  text = (
      f"This network flow shows a connection originating from port {entry['tcp.srcport']} and destined to port {entry['tcp.dstport']}. "
      f"For the HTTP protocol {entry['http.content_length']} bytes of content were transferred. {http_response_dict[entry['http.response']]}. "
      f"For the TCP protocol, the length of the tcp segment was {entry['tcp.len']} bytes. "
      f"The packet's sequence number was {entry['tcp.seq']} and its acknowledgement number was {entry['tcp.ack']} with a raw value of {entry['tcp.ack_raw']}. "
      f"The connection flags were: SYN {entry['tcp.connection.syn']}, RST {entry['tcp.connection.rst']}, FIN {entry['tcp.connection.fin']} and SYN-ACK {entry['tcp.connection.synack']}. "
      f"The overall TCP flags were set to {entry['tcp.flags']}, which signifies {tcp_flags_dict[entry['tcp.flags']]}. "
      f"Regarding the UDP protocol, this flow belonged to the stream {entry['udp.stream']}. "
      f"The MQTT message length was {entry['mqtt.len']} bytes and the MQTT message type was {entry['mqtt.msgtype']}. "
      f"The MQTT connection flags had a value of {entry['mqtt.conflags']} and the header flags were set to {entry['mqtt.hdrflags']}. "
      f"Regarding the ICMP protocol the icmp checksum value was recorded as {entry['icmp.checksum']}."
  )
  record['text']  = text
  record['label'] = entry['label']
  sequences.append(record)
  return
#######APPLY ON EACH ROW AND CONVERT TO DATAFRAME AND SAVE#####
data.apply(encode,axis=1)
sequences = pd.DataFrame(sequences)
sequences.to_csv('/content/drive/MyDrive/Saves/EdgeIIoT_distinct_multi_category_2.csv',index=False,mode='w')
#sequences.to_csv('/content/drive/MyDrive/Saves/EdgeIIoT_distinct_multi_2.csv',index=False,mode='w')
#sequences.to_csv('/content/drive/MyDrive/Saves/EdgeIIoT_distinct_binary_2.csv',index=False,mode='w')
############DIAGNOSTICS##########
sequences.info()
print(sequences.shape)
print(sequences.head(1))