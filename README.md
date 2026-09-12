# Leveraging Language Models in the Detection of Cyberattacks
## Aim of the Project
This project focuses on two known cybersecurity problems, **Log Anomaly Detection** and **Packet Flow Classification**. I showcase the capabilities of language models such as BERT, RoBERTa, Longformer, DistilBERT and BigBird on supervised Log Anomaly Detection and Network Packet Flow classification(binary and multi-class). The prediction results are then analyzed by explainable AI techniques and their outputs visualized accordingly. These visualizations can be very informative to a security expert.
## Experimental setup and workflow
The source code is written in Python using the **HuggingFace** framework. The experiments were performed on an NVIDIA DGX system on A100 40GB GPUs for faster fine-tuning and inference with parallelization capabilities. The following machine learning libraries were used:
- **Scikit-learn**
- **Pytorch**
- **Tensorflow**
- **Transformers**
For data preprocessing, mathematical operations and visualization:
- **Pandas**
- **Numpy**
- **Matplotlib**
- **Seaborn**
Finally for the explainable AI part these libraries were used:
  - **Ferret**
  - **Shap**
  - **Transformers Interpret**
## Project Structure
This repository contains the source code of the experiments done for my final year project/Thesis titled: Leveraging Language Models in the Detection of Cyberattacks. The source code is organized in Preprocessing, Model training and evaluation folders. Additionally, i have included the code necessary for benchmarking my proposed model along with the code for applying and visualizing explainable AI (XAI) methods on the predictions of my proposed model.
