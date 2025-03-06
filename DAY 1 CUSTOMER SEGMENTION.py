import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_excel(r"C:\Users\ravid\OneDrive\Desktop\INDOLIKE INTERNSHIP\DAY 1\online_retail_II.xlsx\online_retail_II.xlsx")
print(df) # print all the data

print(df.head()) # FIRST FIVE
#   Invoice StockCode  ... Customer ID         Country
# 0  489434     85048  ...     13085.0  United Kingdom
# 1  489434    79323P  ...     13085.0  United Kingdom
# 2  489434    79323W  ...     13085.0  United Kingdom
# 3  489434     22041  ...     13085.0  United Kingdom
# 4  489434     21232  ...     13085.0  United Kingdom

df.describe(include="all") # describe data
df.info()
print("*"*100)
df.drop_duplicates(inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)
# Convert the necessary columns to the appropriate data types if needed
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Drop any irrelevant columns if necessary (e.g., 'Country' if not needed)
df = df.drop(['Country'], axis=1)

# Create a 'TotalPrice' column
df['TotalPrice'] = df['Quantity'] * df['Price']

# Group by CustomerID to get the total amount spent, number of orders, and average price
customer_data = df.groupby('Customer ID').agg({
    'TotalPrice': 'sum',
    'Invoice': 'nunique',
    'Quantity': 'sum'
}).rename(columns={
    'TotalPrice': 'TotalSpent',
    'Invoice': 'NumOrders ',
    'Quantity': 'TotalQuantity'
})

# Reset index
customer_data = customer_data.reset_index()

# Display the customer_df
customer_data.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Plot the distribution of TotalSpent
plt.figure(figsize=(10, 6))
sns.histplot(customer_data['TotalSpent'], kde=True)
plt.title('Distribution of Total Spent')
plt.show()

# Plot the relationship between TotalSpent and NumOrders
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_data, x='NumOrders ', y='TotalSpent')
plt.title('Total Spent vs. Number of Orders')
plt.show()

# Pairplot to see relationships between features
sns.pairplot(customer_data)
plt.show()

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Normalize the data
scaled_features = scaler.fit_transform(customer_data[['TotalSpent', 'NumOrders ', 'TotalQuantity']])

# Convert scaled features back to DataFrame
scaled_data = pd.DataFrame(scaled_features, columns=['TotalSpent', 'NumOrders ', 'TotalQuantity'])

from sklearn.cluster import KMeans

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Apply K-Means with the optimal number of clusters (e.g., 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_data, x='NumOrders ', y='TotalSpent', hue='Cluster', palette='viridis')
plt.title('Customer Segmentation')
plt.show()

# Group by cluster to see the average characteristics of each cluster
cluster_summary = customer_data.groupby('Cluster').agg({
    'TotalSpent': 'mean',
    'NumOrders ': 'mean',
    'TotalQuantity': 'mean',
    'Customer ID': 'count'
}).rename(columns={'Customer ID': 'NumCustomers'})

