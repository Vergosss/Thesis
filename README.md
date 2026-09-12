# Leveraging Language Models in the Detection of Cyberattacks
## Aim of the Project
This project focuses on two known cybersecurity problems, **Log Anomaly Detection** and **Packet Flow Classification**. I showcase the capabilities of language models such as BERT, RoBERTa, Longformer, DistilBERT and BigBird on supervised **Log Anomaly Detection** and **Network Packet Flow classification** (binary and multi-class). The prediction results are then analyzed by explainable AI techniques and their outputs visualized accordingly. These visualizations can be very informative to a security expert.
## Experimental Setup and Workflow
The source code is written in Python using the **HuggingFace** framework. The experiments were performed on an NVIDIA DGX system on A100 40GB GPUs for faster fine-tuning and inference with parallelization capabilities. The following machine learning libraries were used:<br>
  - **Scikit-learn**
  - **Pytorch**
  - **Tensorflow**
  - **Transformers**
    
For data preprocessing, mathematical operations and visualization:<br>
  - **Pandas**
  - **Numpy**
  - **Matplotlib**
  - **Seaborn**
    
Finally for the explainable AI part these libraries were used:<br>
   - **Ferret**
  - **Shap**
  - **Transformers Interpret**<br>


The Log Anomaly Detection experiments were conducted on the **HDFS** dataset:<br>
  - https://github.com/logpai/loghub/blob/master/HDFS/README.md

While for the Network Flow Classification task, the models were evaluated on two famous packet flow datasets:<br>
  - IoT-23 https://www.stratosphereips.org/datasets-iot23
  - Edge-IIoTset https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications#files
  
For the Log Anomaly Detection task BERT, DistilBERT, RoBerta, ALBERT, Longformer and Bigbird were compared on the Binary classification of log sequences into Benign or Anomaly category. The most lightweight model (DistilBERT) was then used for the classification of network packet flows into Benign-Malicious categories (binary classification) and into different attack types (Multi-Class Classification). A flow preprocessing technique is used, converting them into a textual form so DistilBERT can handle them. The model is compared with various machine learning models and neural networks such as XGBoost, Decision Trees, Random Forest and MultiLayer Perceptron. DistilBERT's predictions on both tasks were analyzed by the explainable AI techniques SHAP and Integrated Gradients and these two were compared using the Sufficiency, Comprehensiveness and Correlation with Leave One Out scores. Finally, for the Log Anomaly Detection task, the SHAP explanations were visualized showing the influence of each token-word on classifying its sequence as Anomalous. These visualizations can aid a security expert when analyzing a threat. In order to achieve higher speeds for real-time inference, multiple gpus are leveraged using data parallelism. Each execution scenario is benchmarked using **Global Latency**, **Throughput**, **Average Step Latency** and **Average Sample Latency** metrics.
 
## Project Structure
The repository follows a clear folder structure comprising of the folders:<br>
  - **preprocessing**
  - **classifiers**
  - **explainability**
  - **Benchmarking**

**Preprocessing** includes the dataset preprocessing code for the Network Flow Classification problem. **Classifiers** include the code implementing the main models of the thesis with seperate files for language models, classic machine learning algorithms and neural networks. **Explainability** contains the code implementing SHAP and Integrated Gradients explainable AI methods, applying them to my model's predictions then visualizing them for both tasks. At last the **Benchmarking** folder includes the python code and shell scripts for executing parallel inference on multiple gpus. 
