# shopper_spectrum.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\online_retail.csv", encoding='ISO-8859-1')


# ===============================
# STEP 1: Data Cleaning
# ===============================
df = df.dropna(subset=['CustomerID', 'Description'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(str)

# ===============================
# STEP 2: RFM Feature Engineering
# ===============================
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# ===============================
# STEP 3: Clustering (KMeans)
# ===============================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Optional: view RFM cluster output
print("\n--- RFM Clusters ---")
print(rfm.head())

# ===============================
# STEP 4: Collaborative Filtering
# ===============================
user_item_matrix = df.pivot_table(index='CustomerID',
                                  columns='StockCode',
                                  values='Quantity',
                                  aggfunc='sum').fillna(0)

# Normalize each user's item interaction
user_item_matrix_norm = user_item_matrix.div(user_item_matrix.max(axis=1), axis=0)

# Compute similarity between users
user_similarity = user_item_matrix_norm.dot(user_item_matrix_norm.T)

# Recommendation function
def recommend_items(customer_id, top_n=5):
    if customer_id not in user_similarity:
        return f"Customer ID {customer_id} not found."
    similar_users = user_similarity[customer_id].sort_values(ascending=False)[1:6]
    recommended_items = user_item_matrix.loc[similar_users.index].mean().sort_values(ascending=False).head(top_n)
    return recommended_items

# ===============================
# STEP 5: Try a Sample Recommendation
# ===============================
sample_customer = user_item_matrix.index[0]
recommendation = recommend_items(sample_customer)

print(f"\n--- Recommendations for Customer {sample_customer} ---")
print(recommendation)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# RFM Feature Engineering
snapshot_date = df_cleaned['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df_cleaned.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Standardize RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Collaborative Filtering
user_item_matrix = df_cleaned.pivot_table(index='CustomerID',
                                          columns='StockCode',
                                          values='Quantity',
                                          aggfunc='sum').fillna(0)

# Normalize matrix per user
user_item_matrix_norm = user_item_matrix.div(user_item_matrix.max(axis=1), axis=0)

# Compute similarity
user_similarity = user_item_matrix_norm.dot(user_item_matrix_norm.T)

# Function to recommend items
def recommend_items(customer_id, top_n=5):
    similar_users = user_similarity[customer_id].sort_values(ascending=False)[1:6]
    recommended_items = user_item_matrix.loc[similar_users.index].mean().sort_values(ascending=False).head(top_n)
    return recommended_items

# Example usage
sample_customer = user_item_matrix.index[0]
recommendations = recommend_items(sample_customer)

print("RFM with Clusters:\n", rfm.head())
print("\nRecommendations for Customer ID", sample_customer, ":\n", recommendations)

import pandas as pd

# Load the dataset
df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')

# Drop rows with missing CustomerID or Description
df_cleaned = df.dropna(subset=['CustomerID', 'Description'])

# Filter out rows with non-positive Quantity or UnitPrice
df_cleaned = df_cleaned[(df_cleaned['Quantity'] > 0) & (df_cleaned['UnitPrice'] > 0)]

# Convert InvoiceDate to datetime
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

# Create a new column 'TotalPrice'
df_cleaned['TotalPrice'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']

# Convert CustomerID to string
df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(str)

# Preview cleaned data
print(df_cleaned.info())
print(df_cleaned.head())
