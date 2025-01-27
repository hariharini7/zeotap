import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class CustomerLookalikeModel:
    def __init__(self):
        self.feature_matrix = None
        self.customer_ids = None
        self.scaler = StandardScaler()
        
    def _calculate_customer_features(self, transactions_df, customers_df, products_df):
        """
        Calculate features for each customer based on their profile and transaction history
        """
        # Debug info before merge
        print("\nDebug Info - Before Merge:")
        print(f"Products columns: {products_df.columns}")
        print(f"Transactions columns: {transactions_df.columns}")
        
        # Merge transactions with product information
        trans_prod = transactions_df.merge(products_df, on='ProductID', how='left')
        
        # Debug info after merge
        print("\nDebug Info - After Merge:")
        print(f"Merged columns: {trans_prod.columns}")
        print(f"Sample merged data:\n{trans_prod.head()}")
        
        # Use the correct 'Price' column from products_df
        if 'Price_y' not in trans_prod.columns:
            raise KeyError("The 'Price_y' column is missing after merging transactions and products DataFrames.")
        
        if trans_prod['Price_y'].isna().any():
            print("\nWarning: Some prices are missing after merge!")
            missing_products = transactions_df[
                ~transactions_df['ProductID'].isin(products_df['ProductID'])
            ]['ProductID'].unique()
            print(f"Products missing from products_df: {missing_products}")
        
        # Calculate transaction-based features using 'Price_y'
        transaction_features = trans_prod.groupby('CustomerID').agg({
            'TransactionID': 'count',  # Number of transactions
            'TotalValue': ['sum', 'mean', 'std'],  # Spending patterns
            'Quantity': ['sum', 'mean', 'std'],  # Purchase quantity patterns
            'Price_y': ['mean', 'std'],  # Price preference
            'ProductID': 'nunique'  # Product variety
        }).fillna(0)
        
        # Flatten column names
        transaction_features.columns = [
            'transaction_count',
            'total_spend',
            'avg_transaction_value',
            'std_transaction_value',
            'total_quantity',
            'avg_quantity',
            'std_quantity',
            'avg_price_preference',
            'std_price_preference',
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
        
        # Calculate recency and frequency
        latest_date = transactions_df['TransactionDate'].max()
        customer_recency = transactions_df.groupby('CustomerID')['TransactionDate'].max()
        customer_recency = (latest_date - customer_recency).dt.days
        
        # Calculate customer age
        customer_age = (latest_date - customers_df['SignupDate']).dt.days
        
        # Create region dummies
        region_dummies = pd.get_dummies(customers_df['Region'], prefix='region')
        
        # Combine all features
        features = pd.concat([
            transaction_features,
            category_pivot,
            customer_recency.rename('recency'),
            customer_age.rename('customer_age'),
            region_dummies
        ], axis=1).fillna(0)
        
        return features
    
    def fit(self, transactions_df, customers_df, products_df):
        """
        Prepare the model by calculating features and scaling them
        """
        # Calculate features
        features = self._calculate_customer_features(transactions_df, customers_df, products_df)
        
        # Store customer IDs
        self.customer_ids = features.index.tolist()
        
        # Scale features
        self.feature_matrix = self.scaler.fit_transform(features)
        
        return self
    
    def get_similar_customers(self, customer_id, n_recommendations=3):
        """
        Find similar customers for a given customer ID
        """
        if customer_id not in self.customer_ids:
            raise ValueError(f"Customer {customer_id} not found in the dataset")
        
        # Get customer index
        customer_idx = self.customer_ids.index(customer_id)
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity([self.feature_matrix[customer_idx]], self.feature_matrix)[0]
        
        # Get top similar customers (excluding self)
        similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
        
        # Prepare recommendations
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'customer_id': self.customer_ids[idx],
                'similarity_score': similarity_scores[idx]
            })
        
        return recommendations

def generate_lookalike_recommendations():
    """
    Generate lookalike recommendations for customers C0001-C0020
    """
    # Load data
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert date columns
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    # Initialize and fit the model
    model = CustomerLookalikeModel()
    model.fit(transactions_df, customers_df, products_df)
    
    # Generate recommendations for C0001-C0020
    recommendations = {}
    for i in range(1, 21):
        customer_id = f'C{i:04d}'
        try:
            similar_customers = model.get_similar_customers(customer_id)
            recommendations[customer_id] = [
                {
                    'similar_customer_id': rec['customer_id'],
                    'similarity_score': round(rec['similarity_score'], 4)
                }
                for rec in similar_customers
            ]
        except ValueError as e:
            print(f"Warning: {e}")
    
    # Create output DataFrame
    output_rows = []
    for customer_id, recs in recommendations.items():
        row = {
            'customer_id': customer_id,
            'recommendations': str([
                f"{rec['similar_customer_id']}:{rec['similarity_score']}"
                for rec in recs
            ])
        }
        output_rows.append(row)
    
    output_df = pd.DataFrame(output_rows)
    
    # Save to CSV
    output_df.to_csv('Lookalike.csv', index=False)
    
    return recommendations

if __name__ == "__main__":
    recommendations = generate_lookalike_recommendations()
    
    # Print sample recommendations
    print("\nSample Recommendations:")
    for customer_id, recs in list(recommendations.items())[:5]:
        print(f"\nCustomer {customer_id}:")
        for idx, rec in enumerate(recs, 1):
            print(f"  {idx}. Customer {rec['similar_customer_id']} "
                  f"(Similarity: {rec['similarity_score']:.4f})")