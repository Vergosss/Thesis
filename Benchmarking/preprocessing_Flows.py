import pandas as pd
from datasets import Dataset
import time
start = time.perf_counter()
sequences = pd.read_csv('/storage/data2/up1072604/data/IoT23_sequences_multi.csv')
###shuffle###
sequences = sequences.sample(frac=1,random_state=42)

###Make sure labels are integers not floats###
sequences['label'] = sequences['label'].astype(int)
###Information-Dimensions-Columns-Label Distribution-Unique labels###

sequences.info()
print('Dimensions: ',sequences.shape)
print('Columns: ',sequences.columns)
print('Label distribution:',sequences['label'].value_counts())
print('Unique labels: ',sequences['label'].unique())

###Dataframe to HuggingFace Dataset###

sequences = Dataset.from_pandas(sequences)
###Tokenizer###
tokenizer = AutoTokenizer.from_pretrained('/storage/data2/up1072604/saved_tokenizers/IoT23/multi')

def tokenize_logs(entry):
  tokens = tokenizer(entry['text'],padding='max_length',truncation=True)
  tokens['labels'] = entry['label']
  return tokens
  
###TOKENIZING DATASET###

sequences= sequences.map(tokenize_logs,batched=True)
print(sequences.sample(1))
sequences.save_to_disk('/storage/data2/up1072604/data/tokenized_IoT23_multi')
end = time.perf_counter()
print(f'Time for preprocessing (Data Loading,conversion,tokenizing and saving) - Preprocessing Latency: {start-end:.2f}') 
print(f'How many samples per second can this pipeline handle - Preprocessing Throughput: {len(sequences)/(start-end):.2f}')

