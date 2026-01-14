import pandas as pd
from sklearn.model_selection import train_test_split


japan_df = pd.read_excel("JPN Data (1).xlsx")

japan_df.columns = (
    japan_df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.upper()
)
japan_df['AGE_CAR'] = pd.to_datetime(japan_df['AGE_CAR'], errors='coerce')
japan_df['AGE_CAR'].fillna(pd.Timestamp("2019-01-01"), inplace=True)
reference_date = pd.Timestamp("2019-07-01")
japan_df['AGE_CAR_DAYS'] = (reference_date - japan_df['AGE_CAR']).dt.days

def maint_segment(days):
    if days < 200:
        return 1
    elif days < 360:
        return 2
    elif days < 500:
        return 3
    else:
        return 4
#Create MAINT_SEGMENT
japan_df['MAINT_SEGMENT'] = japan_df['AGE_CAR_DAYS'].apply(maint_segment)

#Select Features and Target
X = japan_df[['CURR_AGE', 'GENDER', 'ANN_INCOME', 'MAINT_SEGMENT']]
y = japan_df['PURCHASE']

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)


#Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

#Verify Split
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

y = y.astype(int)
X = X.fillna(X.median(numeric_only=True))
