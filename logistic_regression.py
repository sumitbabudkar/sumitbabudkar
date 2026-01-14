import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =================================================
# 1. LOAD JAPANESE DATA (TRAINING)
# =================================================
japan_df = pd.read_excel("JPN Data (1).xlsx")

# Standardize column names
japan_df.columns = (
    japan_df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.upper()
)

# -------------------------------------------------
# Encode GENDER (Japan)
# -------------------------------------------------
japan_df['GENDER'] = japan_df['GENDER'].map({
    'M': 1,
    'F': 0,
    'MALE': 1,
    'FEMALE': 0
})
japan_df['GENDER'].fillna(japan_df['GENDER'].mode()[0], inplace=True)

# -------------------------------------------------
# Convert maintenance date → days
# -------------------------------------------------
japan_df['AGE_CAR'] = pd.to_datetime(japan_df['AGE_CAR'], errors='coerce')
japan_df['AGE_CAR'].fillna(pd.Timestamp("2019-01-01"), inplace=True)

reference_date = pd.Timestamp("2019-07-01")
japan_df['AGE_CAR_DAYS'] = (reference_date - japan_df['AGE_CAR']).dt.days

# -------------------------------------------------
# Maintenance segmentation
# -------------------------------------------------
def maint_segment(days):
    if days < 200:
        return 1
    elif days < 360:
        return 2
    elif days < 500:
        return 3
    else:
        return 4

japan_df['MAINT_SEGMENT'] = japan_df['AGE_CAR_DAYS'].apply(maint_segment)

# -------------------------------------------------
# Feature matrix & target
# -------------------------------------------------
X = japan_df[['CURR_AGE', 'GENDER', 'ANN_INCOME', 'MAINT_SEGMENT']]
y = japan_df['PURCHASE'].astype(int)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =================================================
# 2. TRAIN LOGISTIC REGRESSION MODEL
# =================================================
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# =================================================
# 3. MODEL EVALUATION (JAPAN DATA)
# =================================================
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

print("\nMODEL PERFORMANCE (JAPAN DATA)")
print("--------------------------------")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall   :", recall_score(y, y_pred))
print("F1 Score :", f1_score(y, y_pred))
print("ROC-AUC  :", roc_auc_score(y, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

# =================================================
# 4. COEFFICIENT INTERPRETATION
# =================================================
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nMODEL COEFFICIENTS")
print("------------------")
print(coef_df)

# =================================================
# 5. APPLY MODEL TO INDIAN DATA
# =================================================
india_df = pd.read_excel("IN_Data (1).xlsx")

# Standardize columns
india_df.columns = (
    india_df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.upper()
)

# Rename maintenance column
india_df.rename(columns={'DT_MAINT': 'AGE_CAR'}, inplace=True)

# -------------------------------------------------
# Encode GENDER (India)  IMPORTANT
# -------------------------------------------------
india_df['GENDER'] = india_df['GENDER'].map({
    'M': 1,
    'F': 0,
    'MALE': 1,
    'FEMALE': 0
})
india_df['GENDER'].fillna(india_df['GENDER'].mode()[0], inplace=True)

# -------------------------------------------------
# Convert maintenance date → days
# -------------------------------------------------
india_df['AGE_CAR'] = pd.to_datetime(india_df['AGE_CAR'], errors='coerce')
india_df['AGE_CAR'].fillna(pd.Timestamp("2019-01-01"), inplace=True)

india_df['AGE_CAR_DAYS'] = (reference_date - india_df['AGE_CAR']).dt.days
india_df['MAINT_SEGMENT'] = india_df['AGE_CAR_DAYS'].apply(maint_segment)

# -------------------------------------------------
# Prediction on Indian data
# -------------------------------------------------
X_india = india_df[['CURR_AGE', 'GENDER', 'ANN_INCOME', 'MAINT_SEGMENT']]

india_df['PURCHASE_PROBABILITY'] = model.predict_proba(X_india)[:, 1]
india_df['PREDICTED_PURCHASE'] = (india_df['PURCHASE_PROBABILITY'] >= 0.5).astype(int)

# =================================================
# 6. INDIAN MARKET OUTPUT
# =================================================
total_customers = india_df.shape[0]
predicted_buyers = india_df['PREDICTED_PURCHASE'].sum()

print("\nINDIAN MARKET PREDICTION")
print("------------------------")
print("Total Customers :", total_customers)
print("Predicted Buyers:", predicted_buyers)
print("Conversion Rate :", predicted_buyers / total_customers)

# Save predictions
india_df.to_csv("predictions_india.csv", index=False)

print("\nSTEP 4 COMPLETED SUCCESSFULLY ")
