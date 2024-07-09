import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans, AffinityPropagation, Birch, MeanShift, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# download data from https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# Read the data from 'online_retail_II.xlsx' file
tran_df = pd.read_excel('online_retail_II.xlsx')

# Define conditions for data cleaning
c1 = (tran_df['Invoice'].isnull() == False)
c2 = (tran_df['Quantity'] > 0)
c3 = (tran_df['Customer ID'].isnull() == False)
c4 = (tran_df['StockCode'].isnull() == False)
c5 = (tran_df['Description'].isnull() == False)

# Apply data cleaning conditions
tran_df = tran_df[c1 & c2 & c3 & c4 & c5]

# Define columns for duplicate removal
grp = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate']

# Remove duplicated rows based on defined columns
tran_df = tran_df.drop_duplicates(grp)

# Create a new column 'transaction_date' with the date portion of 'InvoiceDate'
tran_df['transaction_date'] = tran_df['InvoiceDate'].dt.date

# Count the occurrences of each product description
cats = tran_df['Description'].value_counts().reset_index()

# Select product descriptions with a count of more than 600 occurrences
cats_tops = cats[cats['count'] > 1190]

cats_tops['product'] = 'prod_' + (cats_tops.index + 1).astype(str)

# Create a list of selected product descriptions
pro_lst = list(set(cats_tops['Description']))

# Filter the dataset to include only the selected product descriptions
tran_df_sels = tran_df[tran_df['Description'].isin(pro_lst)]
tran_df_sels.shape

DF = tran_df_sels.groupby(['Customer ID','Description', 'InvoiceDate'])['Quantity'].sum().reset_index()

cats_tops_df = cats_tops[['Description','product']]
DF = pd.merge(DF, cats_tops_df, on = ['Description'])
cols = ['Customer ID','product', 'InvoiceDate', 'Quantity']
df = DF[cols]

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Define products
products = [f'prod_{i}' for i in range(1, 6)]

# Get max date for tenure and recency calculations
max_date = df['InvoiceDate'].max()

# Initialize new DataFrame
new_data = {'Customer ID': df['Customer ID'].unique()}

for product in products:
    prod_data = df[df['product'] == product]
    avg_qty = prod_data.groupby('Customer ID')['Quantity'].mean().rename(f'{product}_avg_Quantity')
    last_purchase = prod_data.groupby('Customer ID')['InvoiceDate'].max().rename(f'{product}_recency')
    first_purchase = prod_data.groupby('Customer ID')['InvoiceDate'].min().rename(f'{product}_tenure')

    new_data[f'{product}_avg_Quantity'] = avg_qty
    new_data[f'{product}_recency'] = (max_date - last_purchase).dt.days
    new_data[f'{product}_tenure'] = (max_date - first_purchase).dt.days

# check data
new_df = pd.DataFrame(new_data).fillna(0)
new_df.dtypes.reset_index()
new_df.isnull().sum().reset_index()

def handle_data(df, features_num, K):
    # Step 1: Range standardization for all numeric fields
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(df[features_num])

    # Step 2: PCA to keep top K components
    pca = PCA(n_components=K)
    pca_features = pca.fit_transform(standardized_features)

    # Step 3: Transform each PCA column into bin #
    binned_features = pd.DataFrame()
    for i in range(K):
        binned_features[f'pca_{i}'] = pd.qcut(pca_features[:, i], q=10, duplicates='drop').codes

    # Clustering algorithms to apply
    algorithms = {
        'Agglomerative': AgglomerativeClustering(),
        'DBSCAN': DBSCAN(),
        'KMeans': KMeans(n_clusters=K),
        'MiniBatchKMeans': MiniBatchKMeans(n_clusters=K),
        'AffinityPropagation': AffinityPropagation(),
        'BIRCH': Birch(n_clusters=K),
        'MeanShift': MeanShift(),
        'OPTICS': OPTICS(),
        'SpectralClustering': SpectralClustering(n_clusters=K),
        'GaussianMixture': GaussianMixture(n_components=K)
    }

    performance = {}
    cluster_labels = {}
    for name, algorithm in algorithms.items():
        if name in ['AffinityPropagation', 'DBSCAN', 'MeanShift', 'OPTICS']:
            try:
                labels = algorithm.fit_predict(binned_features)
            except:
                performance[name] = -1
                continue
        else:
            labels = algorithm.fit_predict(binned_features)

        # Step 5: Check performance by Silhouette Score
        if len(set(labels)) > 1 and len(set(labels)) <= K:
            score = silhouette_score(binned_features, labels)
            performance[name] = score
            cluster_labels[name] = labels
            df[f'cluster_{name}'] = labels
        else:
            performance[name] = -1

    # Select the best clustering algorithm based on silhouette score
    best_algorithm = max(performance, key=performance.get)
    best_labels = cluster_labels[best_algorithm]
    
    # Add the best clustering labels to the original DataFrame
    df['cluster'] = best_labels    
    df['best_algorithm'] = best_algorithm
       
    return df, performance

#Execute
features_num = [col for col in new_df.columns if 'Quantity' in col or 'recency' in col or 'tenure' in col]
K = 3

new_df, performance = handle_data(new_df, features_num, K)
print(new_df.dtypes.reset_index())
print(performance)

new_df['cluster_KMeans'].value_counts()
new_df = new_df.drop(['Customer ID'],axis = 1)
new_df = new_df.reset_index()
new_df.reset_index()[['Customer ID','cluster_KMeans', 'cluster_Agglomerative', 'cluster_BIRCH']]

#performance evaluation: profiling
def eval_clus(new_df, method):
    # Define the cluster membership variable name based on the method
    clus = f'cluster_{method}'
    
    # List of raw variables in new_df
    vars = [col for col in new_df.columns if 'Quantity' in col or 'recency' in col or 'tenure' in col]
    
    # Group by the cluster membership and calculate mean of the raw variables
    df_res = new_df.groupby(clus)[vars].mean().reset_index()
    
    # Add column 'count' that is the number of samples in each cluster
    df_res['count'] = new_df.groupby(clus).size().reset_index(name='count')['count']
    
    # Transform the columns vars into index columns
    for var in vars:
        df_res[var] = df_res[var] / df_res[var].mean()
    
    return df_res

#Execute
result_df = eval_clus(new_df, 'KMeans')
result_df.to_excel('result_perf.xlsx', index= False)

