import pandas as pd
import numpy as np
import json
import os
##
path = os.getcwd()#pou eimaste
#
logs = pd.read_json(path+'\\..\\Data\\pwnjutsu_dataset-system-json-n19.json',lines=True)#diavase os json object kathe line tou arxeiou
#Split the string to remove the node. if the node doesnt exist keep the full raw string
def clean(raw):
  split = raw.split('type')
  if len(split) > 1:
    return 'type'+split[1]
  return raw
#apply the cleaning
logs['raw'] = logs['raw'].apply(clean) # remove the node that is already present in the host column
#Get only linux audit log lines
system_logs_audit = logs[logs['sourcetype'] == 'linux_audit']
system_logs_audit = system_logs_audit.reset_index(drop=True)#drop the index
#save to a file
print(system_logs_audit.index)
#Re indexing
system_logs_audit.index = range(1,len(system_logs_audit)+1)
print('After reindexing: ', system_logs_audit.index)
print(system_logs_audit.loc[1])
system_logs_audit.to_csv(path+'\\..\\Data\\system_logs_audit.log',header=None,index=False,mode='w')