import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def validate_recommendations(transactions_df, customers_df, recommendations):
    """
    Validate the quality of lookalike recommendations
    """
    # Calculate customer metrics for comparison
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean'],
        'Quantity': ['sum', 'mean'],
        'TransactionID': 'count'
    }).fillna(0)
    
    customer_metrics.columns = [
        'total_spend', 'avg_transaction_value',
        'total_quantity', 'avg_quantity',
        'transaction_count'
    ]
    
    # Add customer region
    customer_metrics = customer_metrics.merge(
        customers_df[['CustomerID', 'Region']],
        left_index=True,
        right_on='CustomerID'
    )
    
    # Analyze similarity of recommended pairs
    similarity_analysis = []
    
    for customer_id, recs in recommendations.items():
        # Ensure recs is a list of strings and parse them
        if isinstance(recs, str):
            recs = recs.split(',')  # Split the string into individual recommendations
        
        for rec in recs:
            try:
                similar_id, similarity_score = rec.split(':')
                similarity_score = float(similarity_score)
            except ValueError as e:
                print(f"Error parsing recommendation for customer {customer_id}: {rec} - {e}")
                continue
            
            # Get base customer metrics
            base_metrics = customer_metrics[
                customer_metrics['CustomerID'] == customer_id
            ].iloc[0]
            
            similar_metrics = customer_metrics[
                customer_metrics['CustomerID'] == similar_id
            ].iloc[0]
            
            # Calculate metric differences
            similarity_analysis.append({
                'base_customer': customer_id,
                'similar_customer': similar_id,
                'similarity_score': similarity_score,
                'same_region': base_metrics['Region'] == similar_metrics['Region'],
                'spend_diff_pct': abs(
                    (base_metrics['total_spend'] - similar_metrics['total_spend'])
                    / base_metrics['total_spend']
                ) if base_metrics['total_spend'] > 0 else np.inf,
                'transaction_count_diff_pct': abs(
                    (base_metrics['transaction_count'] - similar_metrics['transaction_count'])
                    / base_metrics['transaction_count']
                ) if base_metrics['transaction_count'] > 0 else np.inf
            })
    
    similarity_df = pd.DataFrame(similarity_analysis)
    
    # Generate validation metrics
    validation_metrics = {
        'avg_similarity_score': similarity_df['similarity_score'].mean(),
        'region_match_rate': similarity_df['same_region'].mean(),
        'avg_spend_diff_pct': similarity_df['spend_diff_pct'].mean(),
        'avg_transaction_count_diff_pct': similarity_df['transaction_count_diff_pct'].mean()
    }
    
    return validation_metrics, similarity_df

def plot_validation_results(similarity_df):
    """
    Create visualizations of recommendation quality
    """
    plt.figure(figsize=(15, 5))
    
    # Similarity score distribution
    plt.subplot(1, 3, 1)
    sns.histplot(data=similarity_df, x='similarity_score', bins=20)
    plt.title('Distribution of Similarity Scores')
    
    # Region matching
    plt.subplot(1, 3, 2)
    sns.barplot(data=similarity_df, y='same_region')
    plt.title('Region Match Rate')
    
    # Spending difference
    plt.subplot(1, 3, 3)
    sns.boxplot(data=similarity_df, y='spend_diff_pct')
    plt.title('Spending Difference Distribution')
    
    plt.tight_layout()
    plt.savefig('lookalike_validation.png')
    plt.close()

if __name__ == "__main__":
    # Load data
    customers_df = pd.read_csv('Customers.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Load recommendations
    recommendations_df = pd.read_csv('Lookalike.csv')
    recommendations = {
        row['customer_id']: eval(row['recommendations'])
        for _, row in recommendations_df.iterrows()
    }
    
    # Validate recommendations
    validation_metrics, similarity_df = validate_recommendations(
        transactions_df, customers_df, recommendations
    )
    
    # Print validation results
    print("\nValidation Metrics:")
    for metric, value in validation_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create validation plots
    plot_validation_results(similarity_df)