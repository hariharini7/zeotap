import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

class CustomerSegmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_matrix = None
        self.features = None
        self.customer_clusters = None
        
    def _prepare_features(self, transactions_df, customers_df, products_df):
        """
        Prepare customer features for clustering
        """
        # Debug info before merge
        print("\nDebug Info - Before Merge:")
        print(f"Products columns: {products_df.columns}")
        print(f"Transactions columns: {transactions_df.columns}")
        
        # Merge transactions with products
        trans_prod = transactions_df.merge(products_df, on='ProductID', how='left')
        
        # Debug info after merge
        print("\nDebug Info - After Merge:")
        print(f"Merged columns: {trans_prod.columns}")
        print(f"Sample merged data:\n{trans_prod.head()}")
        
        # Use the correct 'Price' column from products_df
        if 'Price_y' not in trans_prod.columns:
            raise KeyError("The 'Price_y' column is missing after merging transactions and products DataFrames.")
        
        # Calculate RFM metrics
        latest_date = transactions_df['TransactionDate'].max()
        
        rfm = transactions_df.groupby('CustomerID').agg({
            'TransactionDate': lambda x: (latest_date - x.max()).days,  # Recency
            'TransactionID': 'count',  # Frequency
            'TotalValue': 'sum'  # Monetary
        }).rename(columns={
            'TransactionDate': 'recency',
            'TransactionID': 'frequency',
            'TotalValue': 'monetary'
        })
        
        # Calculate additional metrics
        purchase_behavior = trans_prod.groupby('CustomerID').agg({
            'Quantity': ['sum', 'mean', 'std'],
            'Price_y': ['mean', 'std'],  # Use Price_y for calculations
            'ProductID': 'nunique'
        }).fillna(0)
        
        purchase_behavior.columns = [
            'total_quantity',
            'avg_quantity_per_transaction',
            'quantity_std',
            'avg_price_preference',
            'price_std',
            'unique_products'
        ]
        
        # Calculate category preferences
        category_pivot = pd.pivot_table(
            trans_prod,
            values='Quantity',
            index='CustomerID',
            columns='Category',
            aggfunc='sum',
            fill_value=0
        )
        category_pivot.columns = [f'category_{col}_quantity' for col in category_pivot.columns]
        
        # Customer age
        customer_age = (latest_date - customers_df['SignupDate']).dt.days
        
        # Combine all features
        features = pd.concat([
            rfm,
            purchase_behavior,
            category_pivot,
            customer_age.rename('customer_age')
        ], axis=1).fillna(0)
        
        return features
    
    def find_optimal_clusters(self, features, max_clusters=10):
        """
        Find optimal number of clusters using multiple metrics
        """
        scaled_features = self.scaler.fit_transform(features)
        
        metrics = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            metrics.append({
                'n_clusters': n_clusters,
                'db_score': davies_bouldin_score(scaled_features, clusters),
                'silhouette_score': silhouette_score(scaled_features, clusters)
            })
        
        return pd.DataFrame(metrics)
    
    def fit(self, transactions_df, customers_df, products_df, n_clusters):
        """
        Fit the clustering model
        """
        # Prepare features
        self.features = self._prepare_features(transactions_df, customers_df, products_df)
        
        # Scale features
        self.feature_matrix = self.scaler.fit_transform(self.features)
        
        # Fit KMeans
        self.best_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.customer_clusters = self.best_model.fit_predict(self.feature_matrix)
        
        # Add cluster assignments to features
        self.features['Cluster'] = self.customer_clusters
        
        return self
    
    def get_cluster_profiles(self):
        """
        Generate cluster profiles
        """
        cluster_profiles = []
        
        for cluster in range(len(np.unique(self.customer_clusters))):
            cluster_data = self.features[self.features['Cluster'] == cluster]
            
            profile = {
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Avg_Recency': cluster_data['recency'].mean(),
                'Avg_Frequency': cluster_data['frequency'].mean(),
                'Avg_Monetary': cluster_data['monetary'].mean(),
                'Avg_Quantity': cluster_data['total_quantity'].mean(),
                'Avg_Price_Preference': cluster_data['avg_price_preference'].mean(),
                'Product_Variety': cluster_data['unique_products'].mean()
            }
            
            cluster_profiles.append(profile)
        
        return pd.DataFrame(cluster_profiles)

def create_visualizations(segmentation, metrics_df):
    """
    Create visualizations for the clustering analysis
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Elbow plot for metrics
    plt.subplot(2, 2, 1)
    plt.plot(metrics_df['n_clusters'], metrics_df['db_score'], marker='o', label='Davies-Bouldin')
    plt.plot(metrics_df['n_clusters'], metrics_df['silhouette_score'], marker='o', label='Silhouette')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Clustering Metrics by Number of Clusters')
    plt.legend()
    
    # 2. RFM distribution by cluster
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(
        segmentation.features['recency'],
        segmentation.features['monetary'],
        c=segmentation.features['Cluster'],
        cmap='viridis'
    )
    plt.xlabel('Recency (days)')
    plt.ylabel('Monetary Value')
    plt.title('Customer Segments: Recency vs Monetary Value')
    plt.colorbar(scatter)
    
    # 3. Cluster sizes
    plt.subplot(2, 2, 3)
    cluster_sizes = pd.Series(segmentation.customer_clusters).value_counts().sort_index()
    cluster_sizes.plot(kind='bar')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    plt.title('Cluster Sizes')
    
    # 4. Feature importance
    plt.subplot(2, 2, 4)
    feature_importance = np.abs(segmentation.best_model.cluster_centers_).mean(axis=0)
    feature_importance = pd.Series(
        feature_importance,
        index=segmentation.features.drop('Cluster', axis=1).columns
    ).sort_values(ascending=False)[:10]
    feature_importance.plot(kind='bar')
    plt.xticks(rotation=45)
    plt.title('Top 10 Feature Importance')
    
    plt.tight_layout()
    plt.savefig('clustering_analysis.png')
    plt.close()

def main():
    # Load data
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert date columns
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    # Initialize segmentation
    segmentation = CustomerSegmentation()
    
    # Find optimal number of clusters
    features = segmentation._prepare_features(transactions_df, customers_df, products_df)
    metrics_df = segmentation.find_optimal_clusters(features)
    
    # Select optimal number of clusters (using elbow method)
    optimal_clusters = metrics_df.loc[
        metrics_df['silhouette_score'].diff().abs().idxmax()
    ]['n_clusters']
    
    # Fit model with optimal clusters
    segmentation.fit(transactions_df, customers_df, products_df, int(optimal_clusters))
    
    # Generate cluster profiles
    cluster_profiles = segmentation.get_cluster_profiles()
    
    # Create visualizations
    create_visualizations(segmentation, metrics_df)
    
    # Save results
    cluster_assignments = pd.DataFrame({
        'CustomerID': segmentation.features.index,
        'Cluster': segmentation.features['Cluster']
    })
    
    # Save results
    metrics_df.to_csv('clustering_metrics.csv', index=False)
    cluster_profiles.to_csv('cluster_profiles.csv', index=False)
    cluster_assignments.to_csv('cluster_assignments.csv', index=False)
    
    return segmentation, metrics_df, cluster_profiles

if __name__ == "__main__":
    segmentation, metrics_df, cluster_profiles = main()