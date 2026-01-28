# =====================================================
# PROJECT: ONLINE SHOPPING DATASET
# SUBJECT: DATA SCIENCE
# =====================================================

# ================== IMPORT LIBRARIES ==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

# ================== LOAD DATA ==================
df = pd.read_csv("file.csv")

print("===== DATA LOADED =====")
print(df.head())

# ================== BASIC FUNCTIONS ==================
print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== SAMPLE =====")
print(df.sample(5))

# ================== REMOVE UNNAMED COLUMN ==================
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# ================== A) DATA CLEANING TECHNIQUES ==================

# 1) HANDLING MISSING VALUES
print("\n===== MISSING VALUES BEFORE =====")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

# MEAN
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# MEDIAN (example – not overwriting)
df_median = df.copy()
for col in num_cols:
    df_median[col] = df_median[col].fillna(df_median[col].median())

# DROP (example)
df_drop = df.dropna()

print("\n===== MISSING VALUES AFTER =====")
print(df.isnull().sum())

# 2) HANDLING DUPLICATE RECORDS
print("\nDuplicates before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates after:", df.duplicated().sum())

# 3) DETECTING & DELETING OUTLIERS (IQR METHOD)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) &
            (df[col] <= Q3 + 1.5 * IQR)]

# 4) HANDLING INCONSISTENT DATA
for col in cat_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

print("\n===== DATA CLEANING COMPLETED =====")

# ================== B) DATA TRANSFORMATION TECHNIQUES ==================

# 5) DATA TYPE CONVERSION
if 'Transaction_Date' in df.columns:
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')

# 6) SCALING DATA (STANDARDIZATION)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

# 7) NORMALIZATION (MIN-MAX)
normalizer = MinMaxScaler()
df_normalized = df.copy()
df_normalized[num_cols] = normalizer.fit_transform(df_normalized[num_cols])

# 8) BINNING
if 'Tenure_Months' in df.columns:
    df['Tenure_Group'] = pd.cut(df['Tenure_Months'],
                                bins=3,
                                labels=['Low', 'Medium', 'High'])

# ================== D) EDA ==================

# 9) MEAN, MEDIAN, MODE
print("\n===== MEAN =====")
print(df[num_cols].mean())

print("\n===== MEDIAN =====")
print(df[num_cols].median())

print("\n===== MODE =====")
print(df[num_cols].mode().iloc[0])

# 10) HISTOGRAM
plt.hist(df[num_cols[0]])
plt.title("Histogram")
plt.xlabel(num_cols[0])
plt.ylabel("Frequency")
plt.show()

# 11) BAR PLOT
if len(cat_cols) > 0:
    df[cat_cols[0]].value_counts().plot(kind='bar')
    plt.title("Bar Plot")
    plt.xlabel(cat_cols[0])
    plt.ylabel("Count")
    plt.show()

# 12) SCATTER PLOT
if len(num_cols) > 1:
    plt.scatter(df[num_cols[0]], df[num_cols[1]])
    plt.title("Scatter Plot")
    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])
    plt.show()

# ================== E) FEATURE ENGINEERING ==================

# 13) ONE HOT ENCODING
df_onehot = pd.get_dummies(df, columns=cat_cols)

# 14) DUMMY VARIABLE CREATION
dummy_df = pd.get_dummies(df[cat_cols], drop_first=True)

# 15) LABEL ENCODING
le = LabelEncoder()
df_label = df.copy()
for col in cat_cols:
    df_label[col] = le.fit_transform(df_label[col])

# 16) FEATURE EXTRACTION (DATE)
if 'Transaction_Date' in df.columns:
    df['Year'] = df['Transaction_Date'].dt.year
    df['Month'] = df['Transaction_Date'].dt.month
    df['Day'] = df['Transaction_Date'].dt.day

# 17) FEATURE SCALING
df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

# 18) DIMENSIONALITY REDUCTION (PCA)
df_pca = df[num_cols].copy()
for col in df_pca.columns:
    df_pca[col] = df_pca[col].fillna(df_pca[col].mean())

scaled_data = scaler.fit_transform(df_pca)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
print("\n===== PCA OUTPUT =====")
print(pca_df.head())

# ================== F) TEXT DATA PROCESSING ==================

if 'Product_Description' in df.columns:
    # 18) LOWER CASING
    df['Product_Description'] = df['Product_Description'].astype(str).str.lower()

    # 19) REMOVING PUNCTUATION
    df['Product_Description'] = df['Product_Description'].str.translate(
        str.maketrans('', '', string.punctuation)
    )

    # 20) TOKENIZATION
    df['Tokens'] = df['Product_Description'].str.split()

    print("\n===== TEXT PROCESSING OUTPUT =====")
    print(df[['Product_Description', 'Tokens']].head())

print("\n===== PROJECT EXECUTED SUCCESSFULLY =====")
