# =====================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import os



# -----------------------------
# Load Excel files
# -----------------------------
japan_df = pd.read_excel("JPN Data (1).xlsx")
india_df = pd.read_excel("IN_Data (1).xlsx")

# -----------------------------
# Standardize column names
# -----------------------------
japan_df.columns = (
    japan_df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.upper()
)

india_df.columns = (
    india_df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.upper()
)

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs("outputs/figures", exist_ok=True)

# -----------------------------
# 1. Target Variable Analysis
# -----------------------------
purchase_dist = japan_df['PURCHASE'].value_counts(normalize=True) * 100
print("Purchase Distribution (%):")
print(purchase_dist)

# -----------------------------
# 2. Age vs Purchase
# -----------------------------
plt.figure()
plt.hist(japan_df[japan_df['PURCHASE'] == 1]['CURR_AGE'], bins=15, alpha=0.7, label='Purchased')
plt.hist(japan_df[japan_df['PURCHASE'] == 0]['CURR_AGE'], bins=15, alpha=0.7, label='Not Purchased')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.title('Age Distribution by Purchase')
plt.savefig("outputs/figures/age_vs_purchase.png")
plt.close()

# -----------------------------
# 3. Gender vs Purchase
# -----------------------------
gender_purchase = japan_df.groupby('GENDER')['PURCHASE'].mean()
print("\nPurchase Rate by Gender:")
print(gender_purchase)

# -----------------------------
# 4. Income vs Purchase
# -----------------------------
plt.figure()
plt.boxplot(
    [
        japan_df[japan_df['PURCHASE'] == 0]['ANN_INCOME'],
        japan_df[japan_df['PURCHASE'] == 1]['ANN_INCOME']
    ],
    labels=['Not Purchased', 'Purchased']
)
plt.ylabel('Annual Income')
plt.title('Income vs Purchase')
plt.savefig("outputs/figures/income_vs_purchase.png")
plt.close()

# -----------------------------
# 5. Maintenance Segment vs Purchase
# -----------------------------
if 'MAINT_SEGMENT' in japan_df.columns:
    maint_purchase = japan_df.groupby('MAINT_SEGMENT')['PURCHASE'].mean()
    print("\nPurchase Rate by Maintenance Segment:")
    print(maint_purchase)

    maint_purchase.plot(kind='bar')
    plt.xlabel('Maintenance Segment')
    plt.ylabel('Purchase Probability')
    plt.title('Maintenance Segment vs Purchase')
    plt.savefig("outputs/figures/maintenance_vs_purchase.png")
    plt.close()
else:
    print("\nMAINT_SEGMENT not found — skipping maintenance analysis")

# -----------------------------
# 6. Indian Market Distribution Check
# -----------------------------
required_cols = ['CURR_AGE', 'ANN_INCOME']
if all(col in india_df.columns for col in required_cols):
    print("\nIndian Market Summary:")
    print(india_df[required_cols].describe())

print("\nEDA completed successfully.")
