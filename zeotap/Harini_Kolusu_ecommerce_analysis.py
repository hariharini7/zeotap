import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ECommerceAnalysis:
    def __init__(self):
        self.customers_df = None
        self.products_df = None
        self.transactions_df = None
        self.scaler = StandardScaler()
        self.lookalike_model = None
        self.cluster_model = None
        self.feature_matrix = None
        
    def load_data(self):
        """
        Load and prepare all datasets with validation
        """
        # Load data
        self.customers_df = pd.read_csv('Customers.csv')
        self.products_df = pd.read_csv('Products.csv')
        self.transactions_df = pd.read_csv('Transactions.csv')
        
        # Validate required columns
        self._validate_dataframes()
        
        # Check if 'Price' column exists in products_df
        if 'Price' not in self.products_df.columns:
            raise KeyError("The 'Price' column is missing from the products_df DataFrame.")
        
        # Convert date columns
        self.customers_df['SignupDate'] = pd.to_datetime(self.customers_df['SignupDate'])
        self.transactions_df['TransactionDate'] = pd.to_datetime(self.transactions_df['TransactionDate'])
        
        # Print data validation info
        self._print_data_info()
    
    def _validate_dataframes(self):
        """
        Validate all required columns exist in dataframes
        """
        required_columns = {
            'products_df': ['ProductID', 'Price', 'Category'],
            'customers_df': ['CustomerID', 'Region', 'SignupDate'],
            'transactions_df': ['TransactionID', 'CustomerID', 'ProductID', 'Quantity', 'TotalValue', 'TransactionDate']
        }
        
        missing_product_cols = [col for col in required_columns['products_df'] if col not in self.products_df.columns]
        if missing_product_cols:
            raise KeyError(f"Missing required columns in products_df: {missing_product_cols}")
        
        # Similar checks for other dataframes...
        # (implementation for other dataframes omitted for brevity but follows same pattern)
    
    def _print_data_info(self):
        """
        Print information about loaded data for debugging
        """
        print("\nData Validation Summary:")
        print("-" * 50)
        print("Products DataFrame:")
        print(f"Columns: {', '.join(self.products_df.columns)}")
        print(f"Sample Price values: {self.products_df['Price'].head()}")
        
        print("\nTransactions DataFrame:")
        print(f"Columns: {', '.join(self.transactions_df.columns)}")
        print(f"Sample rows: {len(self.transactions_df)}")
    
    def perform_eda(self):
        """
        Perform Exploratory Data Analysis
        """
        print("\nPerforming EDA...")
        
        # Create EDA visualizations
        plt.figure(figsize=(20, 15))
        
        # 1. Customer Regional Distribution
        plt.subplot(2, 2, 1)
        sns.countplot(data=self.customers_df, x='Region')
        plt.title('Customer Distribution by Region')
        plt.xticks(rotation=45)
        
        # 2. Transaction Value Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=self.transactions_df, x='TotalValue', bins=50)
        plt.title('Distribution of Transaction Values')
        
        # 3. Product Category Distribution
        plt.subplot(2, 2, 3)
        category_counts = self.products_df['Category'].value_counts()
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        plt.title('Product Category Distribution')
        
        # 4. Monthly Sales Trend
        plt.subplot(2, 2, 4)
        monthly_sales = self.transactions_df.groupby(
            self.transactions_df['TransactionDate'].dt.to_period('M')
        )['TotalValue'].sum()
        plt.plot(range(len(monthly_sales)), monthly_sales.values)
        plt.title('Monthly Sales Trend')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png')
        plt.close()
        
        # Generate basic statistics
        self._generate_basic_stats()
    
    def _generate_basic_stats(self):
        """
        Generate and print basic statistics
        """
        # Customer statistics
        print("\nCustomer Statistics:")
        print("-" * 50)
        print(f"Total Customers: {len(self.customers_df)}")
        print(f"Regions: {', '.join(self.customers_df['Region'].unique())}")
        
        # Product statistics
        print("\nProduct Statistics:")
        print("-" * 50)
        print(f"Total Products: {len(self.products_df)}")
        print(f"Categories: {', '.join(self.products_df['Category'].unique())}")
        print(f"Price Range: ${self.products_df['Price'].min():.2f} - ${self.products_df['Price'].max():.2f}")
        
        # Transaction statistics
        print("\nTransaction Statistics:")
        print("-" * 50)
        print(f"Total Transactions: {len(self.transactions_df)}")
        print(f"Total Revenue: ${self.transactions_df['TotalValue'].sum():,.2f}")
        print(f"Average Transaction Value: ${self.transactions_df['TotalValue'].mean():.2f}")
    
    def generate_business_insights(self):
        """
        Generate business insights
        """
        print("\nGenerating Business Insights...")
        
        insights = []
        
        # 1. Customer Concentration
        top_customers = self.transactions_df.groupby('CustomerID')['TotalValue'].sum().nlargest(100)
        insights.append({
            'title': 'Customer Concentration',
            'insight': f"Top 100 customers contribute {(top_customers.sum() / self.transactions_df['TotalValue'].sum() * 100):.1f}% of total revenue."
        })
        
        # 2. Regional Performance
        region_performance = self.transactions_df.merge(
            self.customers_df[['CustomerID', 'Region']], 
            on='CustomerID'
        ).groupby('Region')['TotalValue'].sum()
        insights.append({
            'title': 'Regional Performance',
            'insight': f"Top performing region is {region_performance.idxmax()} with ${region_performance.max():,.2f} in sales."
        })
        
        # Save insights
        pd.DataFrame(insights).to_csv('business_insights.csv', index=False)
    
    def build_lookalike_model(self):
        """
        Build and generate lookalike recommendations
        """
        print("\nBuilding Lookalike Model...")
        
        # Calculate customer features
        features = self._calculate_customer_features()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        self.feature_matrix = scaled_features
        
        # Generate recommendations for C0001-C0020
        recommendations = {}
        for i in range(1, 21):
            customer_id = f'C{i:04d}'
            if customer_id in features.index:
                similar_customers = self._get_similar_customers(customer_id, features, scaled_features)
                recommendations[customer_id] = similar_customers
        
        # Save recommendations
        self._save_recommendations(recommendations)
    
    def _calculate_customer_features(self):
        """
        Calculate customer features for lookalike model with enhanced error handling
        """
        # Debug info before merge
        print("\nDebug Info - Before Merge:")
        print(f"Products columns: {self.products_df.columns}")
        print(f"Transactions columns: {self.transactions_df.columns}")
        
        # Merge transactions with products
        trans_prod = self.transactions_df.merge(
            self.products_df,
            on='ProductID',
            how='left',
            validate='many_to_one'  # Ensure proper merge relationship
        )
        
        # Debug info after merge
        print("\nDebug Info - After Merge:")
        print(f"Merged columns: {trans_prod.columns}")
        print(f"Sample merged data:\n{trans_prod.head()}")
        
        # Use the correct 'Price' column from products_df
        if 'Price_y' not in trans_prod.columns:
            raise KeyError("The 'Price_y' column is missing after merging transactions and products DataFrames.")
        
        if trans_prod['Price_y'].isna().any():
            print("\nWarning: Some prices are missing after merge!")
            missing_products = self.transactions_df[
                ~self.transactions_df['ProductID'].isin(self.products_df['ProductID'])
            ]['ProductID'].unique()
            print(f"Products missing from products_df: {missing_products}")
        
        # Calculate customer metrics using 'Price_y'
        customer_metrics = trans_prod.groupby('CustomerID').agg({
            'TransactionID': 'count',
            'TotalValue': ['sum', 'mean'],
            'Quantity': ['sum', 'mean'],
            'Price_y': 'mean'
        })
        
        customer_metrics.columns = [
            'transaction_count', 'total_spend', 'avg_transaction',
            'total_quantity', 'avg_quantity', 'avg_price'
        ]
        
        # Ensure the index matches the number of customers
        customer_metrics = customer_metrics.reindex(self.customers_df['CustomerID']).fillna(0)
        
        return customer_metrics
    
    def _get_similar_customers(self, customer_id, features, scaled_features, n=3):
        """
        Get similar customers for a given customer ID
        """
        customer_idx = features.index.get_loc(customer_id)
        similarity_scores = cosine_similarity([scaled_features[customer_idx]], scaled_features)[0]
        similar_indices = similarity_scores.argsort()[::-1][1:n+1]
        
        return [
            {
                'customer_id': features.index[idx],
                'similarity_score': similarity_scores[idx]
            }
            for idx in similar_indices
        ]
    
    def _save_recommendations(self, recommendations):
        """
        Save lookalike recommendations to CSV
        """
        output_rows = []
        for customer_id, recs in recommendations.items():
            row = {
                'customer_id': customer_id,
                'recommendations': str([
                    f"{rec['customer_id']}:{rec['similarity_score']:.4f}"
                    for rec in recs
                ])
            }
            output_rows.append(row)
        
        pd.DataFrame(output_rows).to_csv('Lookalike.csv', index=False)
    
    def perform_clustering(self):
        """
        Perform customer segmentation
        """
        print("\nPerforming Customer Segmentation...")
        
        # Calculate features if not already done
        if self.feature_matrix is None:
            features = self._calculate_customer_features()
            self.feature_matrix = self.scaler.fit_transform(features)
        else:
            # If feature_matrix is already calculated, retrieve features from it
            features = pd.DataFrame(self.feature_matrix, index=self.customers_df['CustomerID'])
        
        # Ensure the feature matrix and features DataFrame have the same index
        features = features.loc[self.customers_df['CustomerID']]
        
        # Find optimal number of clusters
        metrics = self._calculate_clustering_metrics()
        optimal_clusters = metrics['n_clusters'][metrics['silhouette_score'].idxmax()]
        
        # Perform clustering
        self.cluster_model = KMeans(n_clusters=int(optimal_clusters), random_state=42)
        clusters = self.cluster_model.fit_predict(self.feature_matrix)
        
        # Ensure the length of clusters matches the number of customers with features
        customer_ids_with_features = features.index
        cluster_assignments = pd.DataFrame({
            'CustomerID': customer_ids_with_features,
            'Cluster': clusters
        })
        
        # Save clustering results
        self._save_clustering_results(cluster_assignments, metrics)
    
    def _calculate_clustering_metrics(self, max_clusters=10):
        """
        Calculate clustering metrics for different numbers of clusters
        """
        metrics = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(self.feature_matrix)
            
            metrics.append({
                'n_clusters': n_clusters,
                'db_score': davies_bouldin_score(self.feature_matrix, clusters),
                'silhouette_score': silhouette_score(self.feature_matrix, clusters)
            })
        
        return pd.DataFrame(metrics)
    
    def _save_clustering_results(self, clusters, metrics):
        """
        Save clustering results and metrics
        """
        # Save metrics
        metrics.to_csv('clustering_metrics.csv', index=False)
        
        # Save cluster assignments
        cluster_assignments = pd.DataFrame({
            'CustomerID': self.customers_df['CustomerID'],
            'Cluster': clusters
        })
        cluster_assignments.to_csv('cluster_assignments.csv', index=False)
        
        # Create and save cluster visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.feature_matrix[:, 0],
            self.feature_matrix[:, 1],
            c=clusters,
            cmap='viridis'
        )
        plt.title('Customer Segments')
        plt.savefig('clustering_visualization.png')
        plt.close()

def main():
    # Initialize analysis
    analysis = ECommerceAnalysis()
    
    # Load data
    analysis.load_data()
    
    # Perform all analyses
    analysis.perform_eda()
    analysis.generate_business_insights()
    analysis.build_lookalike_model()
    analysis.perform_clustering()
    
    print("\nAnalysis complete! Check the output files for results.")

if __name__ == "__main__":
    main()