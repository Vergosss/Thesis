import pandas as pd
import numpy as np
import json
import os
path = os.getcwd()#pou vriskomaste
#
detected_events = pd.read_json(path+'\\..\\Data\\detected_events.json')
#oti parigage to zircolite - ta labels ousiastika
print(detected_events)
#
logs = pd.read_csv(path+'\\..\\Data\\system_logs_audit.log',names=['raw','sourcetype','source','time','host'])#diavase os json object kathe line tou arxeiou
logs['label'] = None
#Re indexing the logs
logs.index = range(1,len(logs)+1)
#Ta logs tou pwnjutsu oste na kano thn antistixisi labels-log lines
##

#metatroph se numpy array ton Stilon(series diladi) tou dataframe
titles = detected_events['title'].to_numpy()
matches = detected_events['matches'].to_numpy()
tags = detected_events['tags'].to_numpy()
title_id = detected_events[['title','id']].to_numpy()#ousiastika ta zeygaronei title kai id
##
tags_matches = detected_events[['tags','matches']].to_numpy()#ousiastika zeygaronei tags matches kai ta vazei os stixia enos array
#meta h to_numpy kanei ena array me ayta ta arrays
##
####Antistixish ton labels sta antistixa logs(log lines)-doyleyei
for record in tags_matches:#to array apo ekso kathe record tou dataframe
    for match in record[1]:#to array apo mesa diladi ta tags tou tag array
        logs['label'][match['row_id']] = record[0]

####

#Print all non null labels
print(logs[logs['label'].notnull()]['raw'])
#
logs[logs['label'].notnull()].to_csv(path+'\\..\\Data\\labeled_logs.csv',mode='w')