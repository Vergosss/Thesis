from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,pipeline
from peft import PeftModel, PeftConfig
import pandas as pd
import numpy as np
import torch
import shap

###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###
config = AutoConfig.from_pretrained("/storage/data2/up1072604/saved_models/HDFS/distilbert")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',config=config)
###
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizers/HDFS/distilbert')
tokenizer = CustomTokenizer(tokenizer)
###
lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_models/HDFS/distilbert')
lora = lora.merge_and_unload()
lora = lora.to(device)
######
print(lora.config.id2label)
print(lora.config.label2id)
print('Num labels:',lora.config.num_labels)
lora.eval()
###
event_traces_test = load_from_disk('/storage/data2/up1072604/data/tokenized_HDFS_test_distilbert')

###Subsample for display###

######
event_traces_test_Anomaly = event_traces_test.filter(lambda x: x['labels'] == 1).sample(n=10,random_state=42)
###check
print(event_traces_test_Anomaly[0])
###Explaining##
model_pipeline = pipeline(task='text-classification',model=model,tokenizer=tokenizer,device=device,return_all_scores=True) #Function that returns propability for the classes
explainer = shap.Explainer(model_pipeline) #build explainer object through pipeline
shaps = explainer(event_traces_test_Anomaly['text'][:]) #generate explanations for the texts
#print(shaps)
single_shap_examples = shap.plots.text(shaps,display=False)

with open('/storage/data2/up1072604/saves/single_shap_examples.html') as f:
    f.write(single_shap_examples,'w')

input('WAIT')
#########
attributed_class = shaps[:, :,"Anomaly"] #select explanations for the Anomaly class only
sum_of_token_attributions_over_samples = shap.plots.bar(shaps[:, :, "Anomaly"].sum(0),display=False)
mean_of_token_attributions_over_samples = shap.plots.bar(shaps[:, :, "Anomaly"].mean(0),display=False)

#
#word to be explained
explaining_word = "error"
explaining_word_attributions = attributed_class._flatten_feature_names()[explaining_word]
#print(explaining_word_attributions)

explaining_word_attributions_per_sample = shap.Explanation(values=np.array(explaining_word_attributions),base_values=np.zeros(len(explaining_word_attributions)),data=[explaining_word]*len(explaining_word_attributions))
explaining_word_across_samples = shap.plots.bar(explaining_word_attributions_per_sample,display=False)

#################SAVING FOR DISPLAY#########
with open('/storage/data2/up1072604/saves/sum_of_token_attributions_over_samples.html') as f:
    f.write(sum_of_token_attributions_over_samples,'w')
###
with open('/storage/data2/up1072604/saves/mean_of_token_attributions_over_samples.html') as f:
    f.write(mean_of_token_attributions_over_samples,'w')
###
with open('/storage/data2/up1072604/saves/explaining_word_across_samples.html') as f:
    f.write(explaining_word_across_samples,'w')