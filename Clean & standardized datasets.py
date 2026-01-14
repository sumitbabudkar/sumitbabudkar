import pandas as pd

japan_df = pd.read_excel("JPN Data (1).xlsx")
india_df = pd.read_excel("IN_Data (1).xlsx")

print(japan_df.head())
print(india_df.head())

#Understand Data Structure
#Check shape & columns
japan_df.shape
india_df.shape

japan_df.columns
india_df.columns

#Data Types Check
japan_df.info()
india_df.info()

#Missing Value Analysis
japan_df.isnull().sum()
india_df.isnull().sum()


print(japan_df.columns.tolist())
print(india_df.columns.tolist())

#STANDARDIZE COLUMN NAMES 
# Clean column names
japan_df.columns = (
    japan_df.columns
    .str.strip()           # remove leading/trailing spaces
    .str.replace(" ", "_") # replace spaces with underscore
    .str.upper()           # make uppercase
)

india_df.columns = (
    india_df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.upper()
)

#Verify Again (Critical)
print(japan_df.columns)
print(india_df.columns)



india_df.rename(columns={'DT_MAINT': 'AGE_CAR'}, inplace=True)



print(japan_df['AGE_CAR'].dtype)
print(india_df['AGE_CAR'].dtype)

reference_date = pd.Timestamp("2019-07-01")


import pandas as pd

# Force datetime conversion (handles bad values safely)
japan_df['AGE_CAR'] = pd.to_datetime(japan_df['AGE_CAR'], errors='coerce')
india_df['AGE_CAR'] = pd.to_datetime(india_df['AGE_CAR'], errors='coerce')


print(japan_df['AGE_CAR'].isna().sum())
print(india_df['AGE_CAR'].isna().sum())

japan_df['AGE_CAR'].fillna(pd.Timestamp("2019-01-01"), inplace=True)
india_df['AGE_CAR'].fillna(pd.Timestamp("2019-01-01"), inplace=True)
reference_date = pd.Timestamp("2019-07-01")

japan_df['AGE_CAR_DAYS'] = (reference_date - japan_df['AGE_CAR']).dt.days
india_df['AGE_CAR_DAYS'] = (reference_date - india_df['AGE_CAR']).dt.days

#Maintenance Segmentation (Now SAFE)
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
india_df['MAINT_SEGMENT'] = india_df['AGE_CAR_DAYS'].apply(maint_segment)
features = ['CURR_AGE', 'GENDER', 'ANN_INCOME', 'MAINT_SEGMENT']
print(japan_df.columns.tolist())


print(japan_df[['AGE_CAR', 'AGE_CAR_DAYS', 'MAINT_SEGMENT']].head())
print(india_df[['AGE_CAR', 'AGE_CAR_DAYS', 'MAINT_SEGMENT']].head())
