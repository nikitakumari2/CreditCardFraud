# Credit Card Fraud Detection

This project focuses on detecting credit card fraud using machine learning techniques and statistical analysis. The dataset used is `creditcard.csv`, which contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

# Dataset 
 
The dataset creditcard.csv contains credit card transactions made over two days in September 2013, with 284,807 transactions (492 fraudulent). Features include:

Time and Amount

28 anonymized features (V1 to V28) from PCA transformation

Binary Class label (0 = legitimate, 1 = fraud)

Class Imbalance:
Legitimate: 284,315 (99.83%)
Fraudulent: 492 (0.17%) 

# Key Findings

1. XGBoost Outperforms Random Forest (Slightly): The XGBoost model generally achieved a higher AUC-ROC (0.8680) and F1-score (0.3333) compared to the Random Forest model (AUC-ROC: 0.6791, F1-Score: 0.0714). This indicates that XGBoost is slightly better at distinguishing between fraudulent and legitimate transactions on this particular test set. However, both models have low recall.

2. High Precision, Low Recall (for both models): Both models has very high precision (especially Random Forest at 1.0000), meaning that when they do predict a transaction as fraudulent, they are very likely to be correct. However, they have very low recall (Random Forest: 0.0370, XGBoost: 0.2222). This means the models are missing a significant number of actual fraudulent transactions. 

3. PySpark Random Forest: AUC-ROC of 0.8390, which shows that using a distributed framework like PySpark can be effective for handling this kind of data.
   
4. T-tests: The t-tests revealed statistically significant differences (p-value < 0.05) in the means of 'Amount', 'V2', 'V3', and 'V4' between fraudulent and legitimate transactions. 'V1' and 'Time' did not show statistically significant differences.
   
5. Chi-Squared Test: The Chi-squared test indicated a statistically significant association between binned 'Amount' values and the 'Class' variable, which suggests that the transaction amount, when categorized into ranges, is related to the likelihood of fraud.

6. KS Test: This test showed statistically significant differences in the distributions of 'Amount' and 'V2' between fraudulent and non-fraudulent transactions. 'V1' and 'Time' did not show significant distributional differences.





