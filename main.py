# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, matthews_corrcoef, precision_recall_curve, auc
import shap
import warnings

warnings.filterwarnings('ignore')

print("Libraries imported successfully.")

# 2. Load Data
try:
    df = pd.read_csv('creditcard.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please place it in the project directory.")
    exit()


# 3. Advanced Feature Engineering
print("Starting feature engineering...")

# Time since last transaction (for simplicity, we sort by Time)
df_sorted = df.sort_values(by='Time')
df_sorted['Time_Diff'] = df_sorted['Time'].diff()
df_sorted['Time_Diff'].fillna(0, inplace=True)

# Rolling average of transaction amount
df_sorted['Amount_Rolling_Avg_5'] = df_sorted['Amount'].rolling(window=5, min_periods=1).mean()

# Reset index and use the engineered dataframe
df = df_sorted.reset_index(drop=True)

print("Feature engineering complete. New features: 'Time_Diff', 'Amount_Rolling_Avg_5'")


# 4. Data Preparation
print("Preparing data for modeling...")

# Scale numerical features
scaler = StandardScaler()
df[['Amount', 'Time', 'Time_Diff', 'Amount_Rolling_Avg_5']] = scaler.fit_transform(df[['Amount', 'Time', 'Time_Diff', 'Amount_Rolling_Avg_5']])

# Define features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE only to the training data
print("Applying SMOTE to the training data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Original training shape: {X_train.shape}")
print(f"SMOTE-resampled training shape: {X_train_smote.shape}")


# 5. Advanced Modeling: Stacked Ensemble
print("\n--- Training Stacked Ensemble Model ---")

# Level 0: Base Models
base_models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'xgboost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1)
}

# Hold-out predictions for meta-model training
meta_train_features = np.zeros((X_train_smote.shape[0], len(base_models)))
meta_test_features = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    print(f"Training {name}...")
    model.fit(X_train_smote, y_train_smote)
    
    # Generate features for the meta-model
    meta_train_features[:, i] = model.predict_proba(X_train_smote)[:, 1]
    meta_test_features[:, i] = model.predict_proba(X_test)[:, 1]

# Level 1: Meta-Model
print("Training meta-model (Logistic Regression)...")
meta_model = LogisticRegression()
meta_model.fit(meta_train_features, y_train_smote)


# 6. Evaluation
print("\n--- Evaluating Stacked Ensemble Model ---")
# Make final predictions
y_pred_stacked = meta_model.predict(meta_test_features)
y_prob_stacked = meta_model.predict_proba(meta_test_features)[:, 1]

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_stacked, target_names=['Non-Fraud', 'Fraud']))

# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_pred_stacked)
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob_stacked)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('precision_recall_curve.png')
print("Precision-Recall curve saved to 'precision_recall_curve.png'")


# 7. Model Interpretability with SHAP
print("\n--- Generating SHAP Summary Plot ---")

# Use the XGBoost model for interpretability
xgb_model = base_models['xgboost']
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Generate and save the SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (XGBoost)")
plt.tight_layout()
plt.savefig('shap_summary.png')
print("SHAP summary plot saved to 'shap_summary.png'")

print("\n--- Project Execution Complete ---")