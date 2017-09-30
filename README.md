# Creditcard Fraud Detection System. 

The detailed analysis of credit card fraudulent data detection system. 

Dataset: [Anonymized credit card transactions labeled as fraudulent or genuine](https://www.kaggle.com/dalpozz/creditcardfraud)
[Download](https://people.rit.edu/~hvp4259/project/data/creditcard.csv)

Algorithm Used:

1. [Simple Logistic regression](https://github.com/hvp004/creditcard-fraud-detection/blob/master/Logistic_Regression.ipynb)
2. [Logistic Regression with Undersampling](https://github.com/hvp004/creditcard-fraud-detection/blob/master/UnderSampling.ipynb)
3. [Synthetic Minority Over Sampling Technique (SMOTE) with gini as the critertionn of best split.](https://github.com/hvp004/creditcard-fraud-detection/blob/master/SMOTE-gini.ipynb) 
4. [Synthetic Minority Over Sampling Technique (SMOTE) with entropy as the critertionn of best split.](https://github.com/hvp004/creditcard-fraud-detection/blob/master/SMOTE--Entropy.ipynb)

Brief Summary of performance. 

| No.  | Model | Precision | Recall | Accuracy | Conclusion |
| ---- | ----- | --------- | ------ | -------- | ---------- |
| 1  | Logistic Regression  | 0.73270 | 0.62398 | 0.99896 | Rejected due to low recall. | 
| 2  | Undersampling & Logistic Regression  | 0.072 | 0.87 | 0.98 | Rejected due to very low precision. | 
| 3  | SMOTE with Gini criteria for best split  | 0.95528 | 0.95528 | 0.99985 | Best model so far. But next model provides better results. Hence, rejected. | 
| 4  | SMOTE with entropy criteria for best split  | 0.96177 | 0.97154 | 0.99988 | Accepted. | 
