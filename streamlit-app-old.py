import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import streamlit as st
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Ensure to set the style for seaborn
sns.set_style("whitegrid")

# Load the data and cache it
@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

data = load_data('C:\\Users\\Lenovo\\AIRFLOW_DOCKER_1\\streamlit\\data\\data_for_model.parquet')
data_expanded = load_data('C:\\Users\\Lenovo\\AIRFLOW_DOCKER_1\\streamlit\\data\\data_expanded_for_model.parquet')

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Home", "Data Overview", "EDA", "Model Configuration", "Results and Visualization"]
page_selection = st.sidebar.radio("Go to", pages)


# Home/Introduction Page
if page_selection == "Home":
    st.title("Market Basket Analysis Application")
    st.subheader("Welcome!")

    st.markdown("""
    This application provides an interactive interface for performing market basket analysis using Apriori and FP-Growth algorithms. It helps you uncover hidden patterns in customer purchases, enabling better product placement, promotions, and inventory management.

    ### Key Features:
    - **Data Exploration:** Get an overview of your transaction data with comprehensive EDA.
    - **Model Building:** Generate association rules using Apriori and FP-Growth models.
    - **Insights and Recommendations:** Discover actionable insights to optimize your business strategies.

    ### How to Use:
    1. **Data Page:** Explore and understand your transaction data.
    2. **Model Configuration:** Set parameters and build your models.
    3. **Results and Insights:** View and analyze the generated rules and insights.

    We hope you find this application useful for gaining valuable insights into your sales data. If you have any questions or need further assistance, please refer to the documentation or contact support.

    **Let's get started!**
    """)

    # Additional resources or links (if any)
    st.markdown("""
    For more information on market basket analysis, visit:
    - [Market Basket Analysis - Wikipedia](https://en.wikipedia.org/wiki/Market_basket_analysis)
    - [Association Rule Learning - Wikipedia](https://en.wikipedia.org/wiki/Association_rule_learning)
    """)

elif page_selection == "Data Overview":
    st.title("Data Overview")
    
    # Display the dataset
    st.subheader("Dataset Preview")
    st.write(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    st.dataframe(data.head())
    st.subheader("Data Collection and Sources")
    st.markdown("""
    The project involves integrating and automating data from multiple sources to streamline Malia Group's billing and collection processes. The data sources encompass a wide range of information, crucial for comprehensive financial reconciliation and analysis.

    **Website ECOM Data**
    - **Rows:** 10,389
    - **Columns:** 9
    - **Description:** This dataset provides a detailed overview of e-commerce orders processed through the website. It includes essential details such as order numbers, shipper names, creation and delivery dates, billing types, amounts, currencies, countries, and airway bills (AWB). These elements offer a complete view of the order lifecycle and financial transactions associated with e-commerce operations.

    **Shipped & Collected - Aramex**
    - **Rows:** 8,810
    - **Columns:** 8
    - **Description:** Focused on shipments handled by Aramex, this dataset includes details like shipper numbers, HAWB (House Air Waybill) numbers, delivery dates, COD (Cash on Delivery) amounts, and invoice dates. It is vital for tracking the status and financial specifics of shipments facilitated by Aramex.

    **Shipped & Collected - Cosmaline**
    - **Rows:** 1,638
    - **Columns:** 8
    - **Description:** Similar to the Aramex dataset, this one details shipments managed by Cosmaline, providing data on shipper numbers, HAWB numbers, delivery dates, COD amounts, and invoice dates. It supports the monitoring of shipping operations and financial reconciliation for Cosmaline's logistics.

    **Collected - Credit Card**
    - **Rows:** 2,205
    - **Columns:** 4
    - **Description:** This dataset tracks credit card payments, capturing details such as order numbers, payment amounts, and payment dates. It plays a crucial role in financial reconciliation, particularly in tracking and verifying credit card transactions.

    **ERP-Oracle Collection**
    - **Rows:** 553
    - **Columns:** 14
    - **Description:** Sourced from the ERP system, this dataset includes information on operating units, order numbers, item types, quantities, and financial data. It serves as a key integration point, linking ERP data with other datasets for a holistic analysis of business operations.

    **Oracle Data**
    - **Rows:** 464
    - **Columns:** 14
    - **Description:** This dataset, akin to the ERP-Oracle Collection, provides detailed information from the Oracle ERP system, including pricing, discounting, and tax data. It is instrumental in analyzing sales performance and financial metrics, offering a deeper understanding of the financial aspects of the business.

    **Daily Rate**
    - **Rows:** 117
    - **Columns:** 2
    - **Description:** Tracking daily exchange rates, this dataset is critical for financial calculations, especially in the context of international transactions involving multiple currencies. Accurate exchange rate data ensures proper conversion and accounting for foreign transactions.

    **Oracle Product Names**
    - **Rows:** 639
    - **Columns:** 3
    - **Description:** This dataset, scraped from the Cosmaline website, includes product names and categories corresponding to the product IDs found in the Oracle data sheet. This additional information enriches the Oracle dataset, enabling better product identification and categorization.

    **Data Acquisition Method:** All datasets, except for the Oracle Product Names, were provided in Excel format and imported into the system from a designated directory. The Oracle Product Names dataset was obtained through a web scraping process from the Cosmaline website, ensuring the most up-to-date and accurate product information.
    """)

elif page_selection == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # Summary Statistics for Numerical Columns
    st.subheader("Summary Statistics for Numerical Columns")
    numerical_cols = ['ordered_quantity', 'unit_selling_price', 'unit_list_price']
    summary_stats = data[numerical_cols].describe()
    st.write(summary_stats)
    
    # Categorical Distribution
    st.subheader("Categorical Distribution")
    def display_categorical_distribution(df, column_name):
        categorical_dist = df[column_name].value_counts()
        st.write(f"Distribution of {column_name}:")
        st.write(categorical_dist)
    
    display_categorical_distribution(data, 'operating_unit_name')  # Replace with actual column name
    
    # Unique Categories Information
    st.subheader("Unique Categories Information")
    def display_unique_categories_info(df, column_name):
        unique_categories = set()
        df[column_name].apply(lambda x: unique_categories.update(x.split(', ')))
        num_unique_categories = len(unique_categories)
        st.write(f"Number of unique categories: {num_unique_categories}")
        st.write(f"Unique categories: {unique_categories}")
    
    display_unique_categories_info(data, 'product_category')  # Replace with actual column name
    
    # Sales Analysis Section
    st.header("Sales Analysis")
    
    # Top 10 Best-Selling Products
    st.subheader("Top 10 Best-Selling Products")
    def plot_top_products(df):
        # Filter out "Product Not Found" and calculate the top 10 best-selling products
        top_products = df[df['product_name'] != 'Product Not Found']\
                        .groupby('product_name')['ordered_quantity']\
                        .sum().sort_values(ascending=False).head(10)
        
        # Plot the top 10 best-selling products
        fig, ax = plt.subplots(figsize=(12, 6))
        top_products.plot(kind='bar', ax=ax)
        ax.set_title('Top 10 Best-Selling Products (Excluding "Product Not Found")')
        ax.set_xlabel('Product Name')
        ax.set_ylabel('Total Quantity Sold')
        ax.set_xticklabels(top_products.index, rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    plot_top_products(data)

    # Monthly Sales Trend
    st.subheader("Monthly Sales Trend")
    def plot_monthly_sales_trend(df):
        # Extract month and year from ordered_date
        df['year_month'] = df['ordered_date'].dt.to_period('M')

        # Aggregate sales by month
        monthly_sales = df.groupby('year_month')['total_sales_with_discount'].sum()

        # Convert the year_month to a string format for better readability
        monthly_sales.index = monthly_sales.index.strftime('%B %Y')

        # Plot the monthly sales trend as a horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_sales.plot(kind='barh', ax=ax)
        ax.set_title('Bar Chart of Monthly Sales')
        ax.set_xlabel('Total Sales with Discount')
        ax.set_ylabel('Month')
        plt.tight_layout()
        st.pyplot(fig)

    plot_monthly_sales_trend(data)

    # Revenue Contribution by Product Category
    st.subheader("Revenue Contribution by Product Category")
    def get_top_categories(df):
        # Group by product_category and sum the ordered_quantity
        category_quantity = df.groupby('product_category')['ordered_quantity'].sum().reset_index()

        # Sort the categories by total quantity sold and select the top 10
        top_categories = category_quantity.sort_values(by='ordered_quantity', ascending=False).head(10)
        
        st.write(top_categories)

    get_top_categories(data_expanded)
    
    # Revenue Contribution by Product Category
    st.subheader("Revenue Contribution by Product Category")
    def plot_category_sales(df):
        category_sales = df.groupby('product_category')['total_sales_with_discount'].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        category_sales.plot(kind='bar', ax=ax)
        ax.set_title('Revenue Contribution by Product Category')
        ax.set_xlabel('Product Category')
        ax.set_ylabel('Total Sales')
        ax.set_xticklabels(category_sales.index, rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    # Order Quantity Distribution
    st.subheader("Order Quantity Distribution")
    def plot_order_quantity_distribution(df):
        # Plot distribution of order quantities
        fig, ax = plt.subplots(figsize=(12, 6))
        df['ordered_quantity'].plot(kind='hist', bins=30, ax=ax)
        ax.set_title('Distribution of Order Quantities')
        ax.set_xlabel('Order Quantity')
        ax.set_ylabel('Frequency')

        # Set the x-axis ticks with increments of 5
        max_order_quantity = df['ordered_quantity'].max()
        ax.set_xticks(range(0, max_order_quantity + 1, 5))

        plt.tight_layout()
        st.pyplot(fig)

    plot_order_quantity_distribution(data)

    # Monetary Distribution
    st.subheader("Monetary Distribution")
    def plot_monetary_distribution(df):
        # Calculate Monetary Value
        monetary = df.groupby('ecom_reference_order_number')['total_sales_with_discount'].sum()
        rfm = pd.DataFrame({'monetary': monetary})

        # Plot the Monetary Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(rfm, x='monetary', kde=True, ax=ax)
        ax.set_title('Monetary Distribution')
        ax.set_xlabel('Total Spend')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)

    plot_monetary_distribution(data)
    
    # Order Analysis Section
    st.header("Order Analysis")
    
    # Recency Distribution
    st.subheader("Recency Distribution")
    def plot_recency_distribution(df):
        df['recency'] = (df['ordered_date'].max() - df['ordered_date']).dt.days

        rfm = pd.DataFrame({
            'recency': df.groupby('ecom_reference_order_number')['recency'].min()
        })

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(rfm, x='recency', kde=True, ax=ax)
        ax.set_title('Recency Distribution')
        ax.set_xlabel('Days Since Last Purchase')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)

    plot_recency_distribution(data)

    # Distribution of Items per Order
    st.subheader("Distribution of Items per Order")
    def save_order_quantity_stats(df):
        order_quantity_stats = df.groupby('ecom_reference_order_number')['ordered_quantity'].sum()
        summary_stats = order_quantity_stats.describe()

        stats = {
            'Statistic': ['Average', 'Minimum', 'Maximum', 'Q1 (25th percentile)', 'Median (50th percentile)', 'Q3 (75th percentile)'],
            'Value': [
                summary_stats['mean'],
                summary_stats['min'],
                summary_stats['max'],
                summary_stats['25%'],
                summary_stats['50%'],
                summary_stats['75%']
            ]
        }
        stats_df = pd.DataFrame(stats)
        st.write(stats_df)

    save_order_quantity_stats(data)

    def plot_items_per_order_distribution(df):
        order_quantity_stats = df.groupby('ecom_reference_order_number')['ordered_quantity'].sum()

        fig, ax = plt.subplots(figsize=(12, 6))
        order_quantity_stats.plot(kind='hist', bins=30, alpha=0.7, ax=ax)
        ax.set_title('Distribution of Items per Order')
        ax.set_xlabel('Number of Items per Order')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)

    plot_items_per_order_distribution(data)
    
    # Pricing and Discount Analysis Section
    st.header("Pricing and Discount Analysis")

    # Distribution of Selling Prices and List Prices
    st.subheader("Distribution of Selling Prices and List Prices")
    def plot_price_distribution(df):
        fig, ax = plt.subplots(figsize=(12, 6))
        df['unit_selling_price'].plot(kind='hist', bins=20, alpha=0.5, label='Selling Price', ax=ax)
        df['unit_list_price'].plot(kind='hist', bins=20, alpha=0.5, label='List Price', ax=ax)
        ax.set_title('Distribution of Selling Prices and List Prices')
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    plot_price_distribution(data)

    # Distribution of Discount Percentages
    st.subheader("Distribution of Discount Percentages")
    def plot_discount_percentage_distribution(df):
        df['discount_percentage'] = (df['unit_list_price'] - df['unit_selling_price']) / df['unit_list_price'] * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        df['discount_percentage'].plot(kind='hist', bins=20, ax=ax)
        ax.set_title('Distribution of Discount Percentages')
        ax.set_xlabel('Discount Percentage')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)

    plot_discount_percentage_distribution(data)

    # Box Plot of Discount Percentages by Product Category
    st.subheader("Box Plot of Discount Percentages by Product Category")
    def plot_discount_percentage_by_category(df):
        df['discount_percentage'] = (df['unit_list_price'] - df['unit_selling_price']) / df['unit_list_price'] * 100
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(x='product_category', y='discount_percentage', data=df, width=0.5, ax=ax)
        ax.set_title('Box Plot of Discount Percentages by Product Category')
        ax.set_xlabel('Product Category')
        ax.set_ylabel('Discount Percentage')
        ax.set_xticklabels(df['product_category'].unique(), rotation=90, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    plot_discount_percentage_by_category(data_expanded)

    # Scatter Plot of Discount Percentage vs. Ordered Quantity
    st.subheader("Scatter Plot of Discount Percentage vs. Ordered Quantity")
    def plot_discount_vs_ordered_quantity(df):
        df['discount_percentage'] = (df['unit_list_price'] - df['unit_selling_price']) / df['unit_list_price'] * 100
        correlation = df['discount_percentage'].corr(df['ordered_quantity'])
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='discount_percentage', y='ordered_quantity', data=df, alpha=0.6, ax=ax)
        ax.set_title('Scatter Plot of Discount Percentage vs. Ordered Quantity')
        ax.set_xlabel('Discount Percentage')
        ax.set_ylabel('Ordered Quantity')
        ax.grid(True)
        ax.text(0.05, max(df['ordered_quantity']) * 0.95, f'Correlation: {correlation:.2f}', fontsize=12, color='red')
        plt.tight_layout()
        st.pyplot(fig)

    plot_discount_vs_ordered_quantity(data)

    # Comparison of Average Discounts for Popular and Less Popular Items
    st.subheader("Comparison of Average Discounts for Popular and Less Popular Items")
    def plot_average_discounts_comparison_hist(df):
        item_popularity = df.groupby('product_name')['ordered_quantity'].sum()
        median_popularity = item_popularity.median()
        df['popularity'] = df['product_name'].map(lambda x: 'Popular' if item_popularity[x] > median_popularity else 'Less Popular')
        average_discounts = df.groupby('product_name')['discount_percentage'].mean()
        df['average_discount'] = df['product_name'].map(average_discounts)

        comparison_df = df[['product_name', 'popularity', 'average_discount']].drop_duplicates()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=comparison_df, x='average_discount', hue='popularity', element='step', stat='density', common_norm=False, bins=20, ax=ax)
        ax.set_title('Comparison of Average Discounts for Popular and Less Popular Items')
        ax.set_xlabel('Average Discount Percentage')
        ax.set_ylabel('Density')
        plt.tight_layout()
        st.pyplot(fig)

    plot_average_discounts_comparison_hist(data)

    # Box Plot of Unit Selling Prices by Top Performing Product Categories
    st.subheader("Box Plot of Unit Selling Prices by Top Performing Product Categories")
    def plot_top_categories_unit_selling_prices(df):
        df['total_sales_with_discount'] = df['ordered_quantity'] * df['unit_selling_price']
        category_sales = df.groupby('product_category')['total_sales_with_discount'].sum()
        top_categories = category_sales.sort_values(ascending=False).head(10).index.tolist()
        top_categories_df = df[df['product_category'].isin(top_categories)]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x='product_category', y='unit_selling_price', data=top_categories_df, order=top_categories, ax=ax)
        ax.set_title('Box Plot of Unit Selling Prices by Top Performing Product Categories')
        ax.set_xlabel('Product Category')
        ax.set_ylabel('Unit Selling Price')
        ax.set_xticklabels(top_categories, rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    plot_top_categories_unit_selling_prices(data_expanded)
    
     # Supply Chain Analysis Section
    st.header("Supply Chain Analysis")

    # Top Distributing Companies by Quantity Shipped
    st.subheader("Top Distributing Companies by Quantity Shipped")
    def plot_top_distributors_treemap(df):
        top_distributors = df.groupby('operating_unit_name')['ordered_quantity'].sum().sort_values(ascending=False).head(10)

        colors = plt.cm.Blues(range(0, 256, int(256/len(top_distributors))))
        colors = colors[::-1]  # Reverse the colors

        fig, ax = plt.subplots(figsize=(12, 6))
        squarify.plot(sizes=top_distributors.values, label=top_distributors.index, alpha=0.8, color=colors, ax=ax)
        ax.set_title('Top Distributing Companies by Quantity Shipped')
        plt.axis('off')  # Remove axes
        plt.tight_layout()
        st.pyplot(fig)

    plot_top_distributors_treemap(data)
    
    # Product Positioning Section
    st.header("Product Positioning")
    
    # Distribution of the Number of Items per Transaction
    st.subheader("Distribution of the Number of Items per Transaction")
    def plot_transaction_diversity(df):
        basket = (df
                .groupby(['ecom_reference_order_number', 'product_name'])['product_name']
                .count().unstack().reset_index().fillna(0)
                .set_index('ecom_reference_order_number'))
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        unique_items_per_transaction = basket_sets.sum(axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(unique_items_per_transaction, bins=range(1, unique_items_per_transaction.max() + 1), edgecolor='k')
        ax.set_title('Distribution of the Number of Items per Transaction')
        ax.set_xlabel('Number of Items per Transaction')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)

    plot_transaction_diversity(data)

    # Distribution of Item Frequencies
    st.subheader("Distribution of Item Frequencies")
    def plot_item_frequencies(df):
        basket = (df
                .groupby(['ecom_reference_order_number', 'product_name'])['product_name']
                .count().unstack().reset_index().fillna(0)
                .set_index('ecom_reference_order_number'))
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        item_frequencies = basket_sets.sum(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(item_frequencies, bins=range(1, item_frequencies.max() + 1), edgecolor='k')
        ax.set_title('Distribution of Item Frequencies')
        ax.set_xlabel('Frequency of Items')
        ax.set_ylabel('Number of Items')
        plt.tight_layout()
        st.pyplot(fig)

    plot_item_frequencies(data)

    # Transaction Diversity Statistics
    st.subheader("Transaction Diversity Statistics")
    def save_transaction_diversity_stats(df):
        basket = (df
                  .groupby(['ecom_reference_order_number', 'product_name'])['product_name']
                  .count().unstack().reset_index().fillna(0)
                  .set_index('ecom_reference_order_number'))
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        unique_items_per_transaction = basket_sets.sum(axis=1)
        stats = unique_items_per_transaction.describe()
        stats_df = pd.DataFrame(stats, columns=['Value']).reset_index()
        stats_df.columns = ['Statistic', 'Value']
        st.table(stats_df)

    save_transaction_diversity_stats(data)

    # Item Frequencies Statistics
    st.subheader("Item Frequencies Statistics")
    def save_item_frequencies_stats(df):
        basket = (df
                  .groupby(['ecom_reference_order_number', 'product_name'])['product_name']
                  .count().unstack().reset_index().fillna(0)
                  .set_index('ecom_reference_order_number'))
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        item_frequencies = basket_sets.sum(axis=0)
        stats = item_frequencies.describe()
        stats_df = pd.DataFrame(stats, columns=['Value']).reset_index()
        stats_df.columns = ['Statistic', 'Value']
        st.table(stats_df)

    save_item_frequencies_stats(data)

    # Combined Niche Products Count
    st.subheader("Combined Niche Products Count")
    def save_combined_niche_products_count(df):
        sales_volume = df.groupby('product_name')['ordered_quantity'].sum().reset_index()
        niche_products = sales_volume[sales_volume['ordered_quantity'] < sales_volume['ordered_quantity'].quantile(0.25)]
        purchase_frequency = df.groupby('product_name')['ecom_reference_order_number'].nunique().reset_index()
        niche_purchase_frequency = purchase_frequency[purchase_frequency['ecom_reference_order_number'] < purchase_frequency['ecom_reference_order_number'].quantile(0.25)]

        combined_niche_products = set(niche_products['product_name']).intersection(
            set(niche_purchase_frequency['product_name'])
        )

        combined_niche_products_df = pd.DataFrame({
            'Product Name': list(combined_niche_products)
        })

        st.table(combined_niche_products_df)

    save_combined_niche_products_count(data)

elif page_selection == "Model Configuration":
    # Code for Model Configuration Page
    # Model Configuration Page
    st.title("Model Configuration")

    # Introduction and Explanation
    st.subheader("Introduction to Market Basket Analysis")
    st.markdown("""
    Market Basket Analysis is a technique used to identify associations or relationships between products. It uses two primary algorithms: **Apriori** and **FP-Growth**.

    ### Key Metrics:
    - **Support:** The proportion of transactions in the dataset that contain the itemset.
    - **Confidence:** A measure of the reliability of the rule. It's calculated as the ratio of the number of transactions containing the itemset to the number of transactions containing the antecedent.
    - **Lift:** The ratio of the observed support to that expected if the two itemsets were independent. A lift greater than 1 indicates a strong association between the antecedent and the consequent.
    """)

    # Model Selection
    st.subheader("Select Model and Set Parameters")
    model_choice = st.radio("Choose the algorithm:", ("Apriori", "FP-Growth"))

    # Common Parameters
    st.markdown("### Common Parameters")
    min_support = st.slider("Minimum Support:", min_value=0.01, max_value=1.0, value=0.02, step=0.01)
    min_confidence = st.slider("Minimum Confidence:", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    min_lift = st.slider("Minimum Lift:", min_value=1.0, max_value=20.0, value=10.0, step=0.5)

    # Function to run the chosen model
    def run_model(df, model_choice, min_support, min_confidence, min_lift):
        basket = (df
                .groupby(['ecom_reference_order_number', 'product_name'])['product_name']
                .count().unstack().reset_index().fillna(0)
                .set_index('ecom_reference_order_number'))
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

        if model_choice == "Apriori":
            frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
        elif model_choice == "FP-Growth":
            frequent_itemsets = fpgrowth(basket_sets, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return None
        
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        filtered_rules = rules[(rules['lift'] >= min_lift)]
        filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        filtered_rules['consequents'] = filtered_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        filtered_rules.reset_index(drop=True, inplace=True)  # Reset the index
        return filtered_rules

    # Run Model Button
    if st.button("Run Model"):
        with st.spinner("Running the model..."):
            result = run_model(data, model_choice, min_support, min_confidence, min_lift)
            if result is None or result.empty:
                st.error("No rules found with the given combination of parameters. Please adjust the thresholds.")
            else:
                st.success(f"Model run successfully! Number of rules: {len(result)}")
                st.write(result)
                st.session_state['results'] = result  # Store results in session state


elif page_selection == "Results and Visualization":
    # Results and Visualizations Page
    st.title("Results and Visualizations")

    # Load the generated rules
    # Assuming 'results' is the DataFrame containing the rules
    if 'results' in st.session_state:
        results = st.session_state['results']

        # Scatter Plot of Support and Confidence
        st.subheader("Scatter Plot of Support and Confidence")
        def plot_support_confidence(result):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=results, x='support', y='confidence', size='lift', hue='lift', sizes=(20, 200), palette='viridis', ax=ax)
            ax.set_title('Scatter Plot of Support vs Confidence')
            ax.set_xlabel('Support')
            ax.set_ylabel('Confidence')
            ax.grid(True)
            st.pyplot(fig)

        plot_support_confidence(results)

        # Filter by Antecedents
        st.subheader("Filter Rules by Antecedents")
        all_antecedents = sorted(set(results['antecedents']))
        selected_antecedents = st.multiselect("Select Antecedents:", all_antecedents)

        if selected_antecedents:
            filtered_results = results[results['antecedents'].isin(selected_antecedents)]
        else:
            filtered_results = results

        # Formatted Display of Rules
        st.subheader("Filtered Rules")
        def print_rules(results):
            if results.empty:
                st.write("No rules found with the selected antecedents.")
            else:
                sorted_results = results.sort_values(by='lift', ascending=False)
                for index, row in sorted_results.iterrows():
                    st.write(f"**Rule {index + 1}:**")
                    st.write(f"  - **Antecedents:** {row['antecedents']}")
                    st.write(f"  - **Consequents:** {row['consequents']}")
                    st.write(f"  - **Support:** {row['support']}")
                    st.write(f"  - **Confidence:** {row['confidence']}")
                    st.write(f"  - **Lift:** {row['lift']}")
                    st.write(f"  - **Leverage:** {row.get('leverage', 'N/A')}")
                    st.write(f"  - **Conviction:** {row.get('conviction', 'N/A')}")
                    st.write(f"  - **Zhang's Metric:** {row.get('zhangs_metric', 'N/A')}")
                    st.write("")

        print_rules(filtered_results)

    else:
        st.write("Please run the model first to generate results.")
