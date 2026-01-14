import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# -------------------------------------------------
# Load Japanese data
# -------------------------------------------------
df = pd.read_excel("JPN Data (1).xlsx")

df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_")
    .str.upper()
)

# Encode Gender
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0, 'MALE': 1, 'FEMALE': 0})
df['GENDER'].fillna(df['GENDER'].mode()[0], inplace=True)

# Date processing
df['AGE_CAR'] = pd.to_datetime(df['AGE_CAR'], errors='coerce')
df['AGE_CAR'].fillna(pd.Timestamp("2019-01-01"), inplace=True)

ref_date = pd.Timestamp("2019-07-01")
df['AGE_CAR_DAYS'] = (ref_date - df['AGE_CAR']).dt.days

def maint_segment(days):
    if days < 200:
        return 1
    elif days < 360:
        return 2
    elif days < 500:
        return 3
    else:
        return 4

df['MAINT_SEGMENT'] = df['AGE_CAR_DAYS'].apply(maint_segment)

# Features & target
X = df[['CURR_AGE', 'GENDER', 'ANN_INCOME', 'MAINT_SEGMENT']]
y = df['PURCHASE'].astype(int)

# -------------------------------------------------
# Train-Test Split (CRITICAL FIX)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Feature Scaling (CRITICAL FIX)
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# Train Logistic Regression
# -------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -------------------------------------------------
# Evaluate on TEST data only
# -------------------------------------------------
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nSTEP 5: THRESHOLD ANALYSIS\n")

for t in [0.3, 0.4, 0.5, 0.6]:
    y_test_pred = (y_test_prob >= t).astype(int)

    print(f"Threshold = {t}")
    print("Accuracy :", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall   :", recall_score(y_test, y_test_pred))
    print("F1 Score :", f1_score(y_test, y_test_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("-" * 40)

# -------------------------------------------------
# Final model decision
# -------------------------------------------------
print("\nRECOMMENDED THRESHOLD: 0.4 (Business Balanced)")
print("STEP 5 COMPLETED SUCCESSFULLY")
