# FedBayNet
Prediction of Chronic Kidney Disease (CKD) using Bayesian Networks through Federated Learning.

## Objective
The primary objective of this project is to develop a robust predictive model for CKD by utilizing BNs to capture and model probabilistic relationships among various risk factors. To address the critical challenge of patient data privacy and scarcity in healthcare machine learning applications, the project integrates a FL approach that enables multiple healthcare institutions to collaboratively train the BN model on decentralized data. This ensures that sensitive patient data is never exposed outside the source institution, minimizing the risk of data breaches and unauthorized access. 

## Dataset
The Kaggle “Risk Factor Prediction of Chronic Kidney Disease” dataset is a medical dataset aimed at predicting the presence of CKD based on various risk factors. It includes 27 attributes drawn from patient health records, of which 11 are binary indicators (e.g., presence of hypertension, diabetes, or anemia) and 16 are categorical or interval-based variables (e.g., specific gravity of urine, serum creatinine, and age). The target variable is the class attribute, a binary indicator specifying whether the patient is affected by CKD or not.

## Project Files
This project uses Python 3.10 and requires the packages listed in the `requirements.txt` file.

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

1. [Data Preprocessing and Exploratory Data Analysis](./Exploratory%20Data%20Analysis)
2. Centralized Bayesian Networks Implementation
   ```
   python main.py --data <path/to/input_data.csv> --output Results
   ```
  
  
