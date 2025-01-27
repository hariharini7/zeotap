import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_customer_metrics(transactions_df, customers_df):
    """
    Calculate key customer metrics for business insights
    """
    # Customer purchase behavior
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'TotalValue': 'sum',
        'Quantity': 'sum',
        'TransactionDate': ['min', 'max']
    }).reset_index()
    
    # Rename columns for clarity
    customer_metrics.columns = ['CustomerID', 'purchase_count', 'total_spend', 
                              'total_quantity', 'first_purchase', 'last_purchase']
    
    # Add customer region
    customer_metrics = customer_metrics.merge(
        customers_df[['CustomerID', 'Region', 'SignupDate']], 
        on='CustomerID'
    )
    
    # Calculate customer lifetime (days)
    customer_metrics['customer_lifetime'] = (
        customer_metrics['last_purchase'] - customer_metrics['first_purchase']
    ).dt.days
    
    # Calculate average purchase value
    customer_metrics['avg_purchase_value'] = (
        customer_metrics['total_spend'] / customer_metrics['purchase_count']
    )
    
    return customer_metrics

def analyze_product_performance(transactions_df, products_df):
    """
    Analyze product performance metrics
    """
    # Merge transactions with products
    product_sales = transactions_df.merge(
        products_df[['ProductID', 'Category', 'Price']], 
        on='ProductID'
    )
    
    # Calculate product metrics
    product_metrics = product_sales.groupby('ProductID').agg({
        'TransactionID': 'count',
        'Quantity': 'sum',
        'TotalValue': 'sum'
    }).reset_index()
    
    # Add product information
    product_metrics = product_metrics.merge(
        products_df, 
        on='ProductID'
    )
    
    # Calculate average purchase quantity
    product_metrics['avg_quantity_per_transaction'] = (
        product_metrics['Quantity'] / product_metrics['TransactionID']
    )
    
    return product_metrics

def analyze_regional_performance(transactions_df, customers_df):
    """
    Analyze performance metrics by region
    """
    # Merge transactions with customer region
    regional_sales = transactions_df.merge(
        customers_df[['CustomerID', 'Region']], 
        on='CustomerID'
    )
    
    # Calculate regional metrics
    regional_metrics = regional_sales.groupby('Region').agg({
        'TransactionID': 'count',
        'CustomerID': 'nunique',
        'TotalValue': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    # Calculate derived metrics
    regional_metrics['avg_transaction_value'] = (
        regional_metrics['TotalValue'] / regional_metrics['TransactionID']
    )
    regional_metrics['avg_customer_value'] = (
        regional_metrics['TotalValue'] / regional_metrics['CustomerID']
    )
    
    return regional_metrics

def analyze_time_trends(transactions_df):
    """
    Analyze sales trends over time
    """
    # Daily sales trends
    daily_sales = transactions_df.groupby('TransactionDate').agg({
        'TotalValue': 'sum',
        'TransactionID': 'count',
        'Quantity': 'sum'
    }).reset_index()
    
    # Calculate moving averages
    daily_sales['moving_avg_value'] = daily_sales['TotalValue'].rolling(
        window=7
    ).mean()
    
    # Monthly aggregation
    monthly_sales = transactions_df.set_index('TransactionDate').resample('M').agg({
        'TotalValue': 'sum',
        'TransactionID': 'count',
        'Quantity': 'sum'
    }).reset_index()
    
    return daily_sales, monthly_sales

def generate_insights(customers_df, products_df, transactions_df):
    """
    Generate comprehensive business insights
    """
    # Calculate all metrics
    customer_metrics = calculate_customer_metrics(transactions_df, customers_df)
    product_metrics = analyze_product_performance(transactions_df, products_df)
    regional_metrics = analyze_regional_performance(transactions_df, customers_df)
    daily_sales, monthly_sales = analyze_time_trends(transactions_df)
    
    # Store insights
    insights = []
    
    # Customer Insights
    top_customers = customer_metrics.nlargest(100, 'total_spend')
    insights.append({
        'title': 'Customer Concentration Risk',
        'insight': f"Top 100 customers contribute {(top_customers['total_spend'].sum() / customer_metrics['total_spend'].sum() * 100):.1f}% of total revenue, suggesting potential concentration risk. Average customer lifetime is {customer_metrics['customer_lifetime'].mean():.1f} days.",
        'metric': 'Revenue Concentration'
    })
    
    # Regional Insights
    top_region = regional_metrics.nlargest(1, 'TotalValue').iloc[0]
    insights.append({
        'title': 'Regional Performance',
        'insight': f"The {top_region['Region']} region leads with ${top_region['TotalValue']:,.2f} in sales and {top_region['CustomerID']} customers, averaging ${top_region['avg_customer_value']:,.2f} per customer.",
        'metric': 'Regional Sales'
    })
    
    # Product Insights
    category_performance = product_metrics.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
    insights.append({
        'title': 'Category Performance',
        'insight': f"Top performing category is {category_performance.index[0]} with ${category_performance.iloc[0]:,.2f} in sales, followed by {category_performance.index[1]} (${category_performance.iloc[1]:,.2f}).",
        'metric': 'Category Sales'
    })
    
    # Time Trend Insights
    mom_growth = monthly_sales['TotalValue'].pct_change().mean() * 100
    insights.append({
        'title': 'Growth Trends',
        'insight': f"Average month-over-month revenue growth is {mom_growth:.1f}%. Peak sales month recorded ${monthly_sales['TotalValue'].max():,.2f} in revenue.",
        'metric': 'Sales Growth'
    })
    
    # Purchase Behavior
    avg_metrics = transactions_df.agg({
        'Quantity': 'mean',
        'TotalValue': 'mean'
    })
    insights.append({
        'title': 'Purchase Behavior',
        'insight': f"Average transaction value is ${avg_metrics['TotalValue']:.2f} with {avg_metrics['Quantity']:.1f} items per transaction. Customer repeat purchase rate is {(customer_metrics['purchase_count'] > 1).mean() * 100:.1f}%.",
        'metric': 'Transaction Metrics'
    })
    
    return insights

def main():
    # Load data
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert date columns
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    # Generate insights
    insights = generate_insights(customers_df, products_df, transactions_df)
    
    # Print insights
    for insight in insights:
        print(f"\n{insight['title']}")
        print("-" * len(insight['title']))
        print(insight['insight'])
        print(f"Key Metric: {insight['metric']}")

if __name__ == "__main__":
    main()