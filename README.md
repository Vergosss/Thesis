# Leveraging Language Models in the Detection of Cyberattacks
## Aim of the Project
This project focuses on two known cybersecurity problems, **Log Anomaly Detection** and **Packet Flow Classification**. I showcase the capabilities of language models such as BERT, RoBERTa, Longformer, DistilBERT and BigBird on supervised Log Anomaly Detection and **Network Packet Flow classification** (binary and multi-class). The prediction results are then analyzed by explainable AI techniques and their outputs visualized accordingly. These visualizations can be very informative to a security expert.
## Experimental setup and workflow
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
For the Log Anomaly Detection task BERT, DistilBERT, RoBerta, ALBERT, Longformer and Bigbird were compared on the Binary classification of log sequences into Benign or Anomaly category. The most lightweight model (DistilBERT) was then used for the classification of network packet flows into Benign-Malicious categories (binary classification) and into different attack types (Multi-Class Classification). A flow preprocessing technique is used, converting them into a textual form so DistilBERT can handle them. The model is compared with various machine learning models and neural networks such as XGBoost, Decision Trees, Random Forest and MultiLayer Perceptron. DistilBERT's predictions on both tasks were analyzed by the explainable AI techniques SHAP and Integrated Gradients and these two were compared using the Sufficiency, Comprehensiveness and Correlation with Leave One Out scores. Finally, for the Log Anomaly Detection task, the SHAP explanations were visualized showing the influence of each token-word on classifying its sequence as Anomalous.
 
## Project Structure
This repository contains the source code of the experiments done for my final year project/Thesis titled: Leveraging Language Models in the Detection of Cyberattacks. The source code is organized in Preprocessing, Model training and evaluation folders. Additionally, i have included the code necessary for benchmarking my proposed model along with the code for applying and visualizing explainable AI (XAI) methods on the predictions of my proposed model.
