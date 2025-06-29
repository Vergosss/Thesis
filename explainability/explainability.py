from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from ferret import Benchmark, IntegratedGradientExplainer, SHAPExplainer
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizer')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
lora = PeftModel.from_pretrained(model,'/storage/data2/up1072604/saved_model')
lora = lora.merge_and_unload()
lora = lora.to(device)
lora.eval()
#
###Explainers###
shap = SHAPExplainer(lora,tokenizer)
integrated_gradients = IntegratedGradientExplainer(lora,tokenizer)
#
bench = Benchmark(model, tokenizer,explainers=[shap,integrated_gradients])
###
sequences_test = pd.read_csv('/storage/data2/up1072604/saves/sequences_test.csv')
#print(len(sequences_test))
#sample = sequences_test.sample()['text'].values[0]
#print(sample)
#input('WAIT')
sequences_test = sequences_test.sample(n=1000,random_state=42)
texts = list(sequences_test['text'])
###
def explain(text):
    #explanations = bench.explain(text,target=1,call_args=arguments) #lets say target=1 positive class
    expl_shap = shap(text,target=1,batch_size=64)
    expl_integrated_gradients = integrated_gradients(text,target=1,internal_batch_size=64)
    explanations = [expl_shap,expl_integrated_gradients]    
    metrics = bench.evaluate_explanations(explanations,target=1)
    return metrics
###Batched Inference######
#print('Metrics for text: \n',explain(sample))
def showTable(object):
            explanation = object.explanation
            return {
            "explainer":explanation.explainer,
            "target":explanation.target,
            "metrics":{score.name:score.score for score in object.evaluation_scores}
            }

#all_metrics = explain(sample)
#print('ALL METRICS: \n',all_metrics[0])
#results = [showTable(example) for example in all_metrics]
#print('\nResults:',results)
shaps = []
igs = []
print(f'length of samples:{len(texts)}')
for text in tqdm(texts):
    #explanations for 2 methods for given text
    expl_shap = shap(text,target=1,batch_size=64)
    expl_integrated_gradients = integrated_gradients(text,target=1,internal_batch_size=64)
    explanations = [expl_shap,expl_integrated_gradients]
    metrics = bench.evaluate_explanations(explanations,target=1) #lista me dyo stixia explanationevaluation tou shap kai ig
    #
    shap_metrics = metrics[0]
    ig_metrics = metrics[1]
    #
    shaps.append({score.name:score.score for score in shap_metrics.evaluation_scores})
    igs.append({score.name:score.score for score in ig_metrics.evaluation_scores})
shaps = pd.DataFrame(shaps)
igs = pd.DataFrame(igs)
print('SHAP METRICS:',shaps.mean(axis=0))
print('Integrated Gradients:',igs.mean(axis=0))
print('OK')
