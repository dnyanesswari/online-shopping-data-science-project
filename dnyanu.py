# =========================================
# ONLINE SHOPPING DATASET – COMPLETE CODE
# As per PDF (All in ONE FRAME)
# =========================================

# ===== IMPORT LIBRARIES =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

# =========================================
# 1. BASIC FUNCTIONS – LOAD DATA
# =========================================

df = pd.read_csv("online_shopping.csv")   # change filename if needed

print("HEAD")
print(df.head())

print("\nINFO")
print(df.info())

print("\nDESCRIBE")
print(df.describe())

print("\nSAMPLE")
print(df.sample())

# =========================================
# 2. DATA CLEANING TECHNIQUES
# =========================================

# ----- HANDLING MISSING VALUES -----

# MEAN
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# MEDIAN (example)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# DROP
df = df.dropna()

# ----- HANDLING DUPLICATE RECORDS -----
df = df.drop_duplicates()

# ----- DETECTING & DELETING OUTLIERS (IQR METHOD) -----
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# ----- HANDLING INCONSISTENT DATA -----
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].str.lower().str.strip()

# =========================================
# 3. DATA TRANSFORMATION TECHNIQUES
# =========================================

# ----- DATA TYPE CONVERSION -----
for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')

# ----- SCALING DATA (STANDARDIZATION) -----
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ----- NORMALIZATION (MIN-MAX) -----
minmax = MinMaxScaler()
df[num_cols] = minmax.fit_transform(df[num_cols])

# ----- BINNING -----
if len(num_cols) > 0:
    df['binned_data'] = pd.cut(df[num_cols[0]],
                               bins=3,
                               labels=['Low', 'Medium', 'High'])

# =========================================
# 4. EDA – EXPLORATORY DATA ANALYSIS
# =========================================

# MEAN, MEDIAN, MODE
print("\nMEAN")
print(df.mean(numeric_only=True))

print("\nMEDIAN")
print(df.median(numeric_only=True))

print("\nMODE")
print(df.mode().iloc[0])

# HISTOGRAM
df[num_cols[0]].hist()
plt.title("Histogram")
plt.show()

# BAR PLOT
df[num_cols[0]].value_counts().plot(kind='bar')
plt.title("Bar Plot")
plt.show()

# SCATTER PLOT
if len(num_cols) > 1:
    plt.scatter(df[num_cols[0]], df[num_cols[1]])
    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])
    plt.title("Scatter Plot")
    plt.show()

# =========================================
# 5. FEATURE ENGINEERING
# =========================================

# ----- ONE HOT ENCODING -----
df_onehot = pd.get_dummies(df, drop_first=True)

# ----- DUMMY VARIABLE CREATION -----
df_dummy = pd.get_dummies(df)

# ----- LABEL ENCODING -----
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ----- FEATURE EXTRACTION -----
X = df.select_dtypes(include=np.number)

# ----- FEATURE SCALING -----
X_scaled = StandardScaler().fit_transform(X)

# ----- DIMENSIONALITY REDUCTION (PCA) -----
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

print("\nPCA DATA")
print(pca_data)

# =========================================
# 6. TEXT DATA PROCESSING TECHNIQUES
# =========================================

# Assume text column name is 'description'
if 'description' in df.columns:

    # LOWER CASING
    df['description'] = df['description'].str.lower()

    # REMOVING PUNCTUATION
    df['description'] = df['description'].apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation))
    )

    # TOKENIZATION
    df['tokens'] = df['description'].apply(lambda x: x.split())

    print("\nTEXT PROCESSING OUTPUT")
    print(df[['description', 'tokens']].head())

# =========================================
# END OF PROJECT CODE
# =========================================
