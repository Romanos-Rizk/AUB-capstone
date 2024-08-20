import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import streamlit as st
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from PIL import Image
import requests
from io import BytesIO

# Set the page layout to wide mode
st.set_page_config(layout="wide")
# Ensure to set the style for seaborn
sns.set_style("whitegrid")

# Load the data and cache it
@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

data = load_data('https://raw.githubusercontent.com/Romanos-Rizk/AUB-capstone/main/data/data_for_model.parquet')
data_expanded = load_data('https://raw.githubusercontent.com/Romanos-Rizk/AUB-capstone/main/data/data_expanded_for_model.parquet')

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Home", "Data Overview", "EDA", "Model Configuration", "Results and Visualization"]
page_selection = st.sidebar.radio("Go to", pages)


# Logo URL
logo_url = "https://raw.githubusercontent.com/Romanos-Rizk/AUB-capstone/main/resources/logos.png"

# Fetch the image data from the URL
response = requests.get(logo_url)
logo = Image.open(BytesIO(response.content))

# Resize the logo image
logo_resized = logo.resize((300, 100))  # Adjust the width and height as needed


# Home/Introduction Page
if page_selection == "Home":
    # Create three columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Leave the first column empty
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image(logo_resized, use_column_width=True)

    with col3:
        # Leave the third column empty
        st.write("")

    st.markdown(
        "<h1 style='font-size:40px;'>AUB x Malia Market Basket Analysis Application</h1>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="font-size:25px;">
        This application provides an interactive interface for performing market basket analysis using Apriori and FP-Growth algorithms. It helps you uncover hidden patterns in customer purchases, enabling better product placement, promotions, and inventory management.
        <br><br>
        <strong>Key Features:</strong>
        <ul>
            <li><strong>Data Exploration:</strong> Get an overview of your transaction data with comprehensive EDA.</li>
            <li><strong>Model Building:</strong> Generate association rules using Apriori and FP-Growth models.</li>
            <li><strong>Insights and Recommendations:</strong> View and analyze the generated rules and insights.</li>
        </ul>
        <br>
        <strong>How to Use:</strong>
        <ol>
            <li><strong>Data Page:</strong> Explore and understand your transaction data.</li>
            <li><strong>Model Configuration:</strong> Set parameters and build your models.</li>
            <li><strong>Results and Insights:</strong> View and analyze the generated rules and insights.</li>
        </ol>
        <br>
        We hope you find this application useful for gaining valuable insights into your sales data. For further assistance, please refer to the documentation or contact support.
        <br><br>
        <strong>Let's get started!</strong>
    </div>
    """, unsafe_allow_html=True)

    # Additional resources or links (if any)
    st.markdown("""
    <div style="font-size:25px;">
        For more information on market basket analysis, visit:
        <ul>
            <li><a href="https://en.wikipedia.org/wiki/Market_basket_analysis" target="_blank">Market Basket Analysis - Wikipedia</a></li>
            <li><a href="https://en.wikipedia.org/wiki/Association_rule_learning" target="_blank">Association Rule Learning - Wikipedia</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        To Malia Group, By AUB Students
    </div>
    """, unsafe_allow_html=True)


elif page_selection == "Data Overview":
        # Create three columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Leave the first column empty
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image(logo_resized, use_column_width=True)

    with col3:
        # Leave the third column empty
        st.write("")
        
    st.title("Data Overview")
    
    # Display the dataset
    st.subheader("Dataset Preview")
    st.write(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    st.dataframe(data.head())

    st.subheader("Data Collection and Sources")
    st.markdown(f"""
    The datasets used in this application provide detailed information about e-commerce order transactions, including the products in each order, the quantities, and the pricing. This data is essential for performing market basket analysis to uncover patterns in customer purchasing behavior.

    **Oracle Data**
    - **Rows:** {data.shape[0]}
    - **Columns:** {data.shape[1]}
    - **Description:** This dataset contains comprehensive details from the Oracle ERP system, including order numbers, product details, quantities, pricing, discounts, and tax data. It provides a complete view of each transaction, necessary for analyzing sales performance and identifying product associations.

    **Oracle Product Names**
    - **Rows:** 639
    - **Columns:** 3
    - **Description:** Sourced from the Cosmaline website, this dataset includes product names and categories associated with the product IDs in the Oracle data. It enriches the primary dataset by offering detailed product information, crucial for understanding the context of each order and facilitating the market basket analysis.
    """)
    
    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        To Malia Group, By AUB Students
    </div>
    """, unsafe_allow_html=True)


elif page_selection == "EDA":
    
        # Create three columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Leave the first column empty
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image(logo_resized, use_column_width=True)

    with col3:
        # Leave the third column empty
        st.write("")
        
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
    
    display_categorical_distribution(data, 'operating_unit_name')
    
    # Sales Analysis Section
    st.header("Sales Analysis")
    
    # Top 10 Best-Selling Products
    st.subheader("Top 10 Best-Selling Products")
    col1, col2 = st.columns([1, 1])  # Create two columns with ratios
    with col1:
        def plot_top_products(df):
            top_products = df[df['product_name'] != 'Product Not Found']\
                            .groupby('product_name')['ordered_quantity']\
                            .sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, 6))
            top_products.plot(kind='bar', ax=ax)
            ax.set_title('Top 10 Best-Selling Products')
            ax.set_xlabel('Product Name')
            ax.set_ylabel('Total Quantity Sold')
            ax.set_xticklabels(top_products.index, rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        plot_top_products(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Top 10 Best-Selling Products' plot shows the products with the highest sales volume. <br><br> This visualization highlights the top-performing products in terms of quantity sold, providing insights into customer preferences and popular items.</p>",
            unsafe_allow_html=True
        )
    # Monthly Sales Trend
    st.subheader("Monthly Sales Trend")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_monthly_sales_trend(df):
            df['year_month'] = df['ordered_date'].dt.to_period('M')
            monthly_sales = df.groupby('year_month')['total_sales_with_discount'].sum()
            monthly_sales.index = monthly_sales.index.strftime('%B %Y')
            fig, ax = plt.subplots(figsize=(6, 4))
            monthly_sales.plot(kind='barh', ax=ax)
            ax.set_title('Monthly Sales Trend')
            ax.set_xlabel('Total Sales with Discount')
            ax.set_ylabel('Month')
            plt.tight_layout()
            st.pyplot(fig)
        plot_monthly_sales_trend(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Monthly Sales Trend' plot displays the total sales revenue for each month. <br><br> It helps in understanding the sales performance over time, identifying peak sales periods, and evaluating seasonal trends or patterns.</p>",
            unsafe_allow_html=True
        )
    
    # Revenue Contribution by Product Category
    st.subheader("Revenue Contribution by Product Category")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_category_sales(df):
            category_sales = df.groupby('product_category')['total_sales_with_discount'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            category_sales.plot(kind='bar', ax=ax)
            ax.set_title('Revenue Contribution by Product Category')
            ax.set_xlabel('Product Category')
            ax.set_ylabel('Total Sales')
            ax.set_xticklabels(category_sales.index, rotation=45, ha='right', fontsize=8)
            ax.grid(False)
            plt.tight_layout()
            st.pyplot(fig)
        plot_category_sales(data_expanded)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Revenue Contribution by Product Category' plot shows the percentage of total revenue generated by each product category. <br><br> It highlights the most financially significant categories and helps identify which ones drive the most revenu</p>",
            unsafe_allow_html=True
        )
    
    # Distribution of Order Quantities
    st.subheader("Distribution of Order Quantities")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_order_quantity_distribution(df):
            fig, ax = plt.subplots(figsize=(6, 4))
            df['ordered_quantity'].plot(kind='hist', bins=30, ax=ax)
            ax.set_title('Distribution of Order Quantities')
            ax.set_xlabel('Order Quantity')
            ax.set_ylabel('Frequency')
            max_order_quantity = df['ordered_quantity'].max()
            ax.set_xticks(range(0, max_order_quantity + 1, 5))
            plt.tight_layout()
            st.pyplot(fig)
        plot_order_quantity_distribution(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Distribution of Order Quantities' plot is a histogram that visualizes the frequency of various order quantities placed by customers. <br><br> It provides insight into purchasing behavior by showing how many items are typically ordered in a single transaction. <br><br> This histogram helps identify common order sizes, revealing whether customers typically buy single items or larger quantities.</p>",
            unsafe_allow_html=True
        )
        
        
     # Monetary Distribution
    st.subheader("Monetary Distribution")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_monetary_distribution(df):
            monetary = df.groupby('ecom_reference_order_number')['total_sales_with_discount'].sum()
            rfm = pd.DataFrame({'monetary': monetary})
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(rfm, x='monetary', kde=True, ax=ax)
            ax.set_title('Monetary Distribution')
            ax.set_xlabel('Total Spend')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
        plot_monetary_distribution(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Monetary Distribution' plot is a histogram that displays the distribution of total spending per order, showing the frequency of different spending levels by customers. <br><br> It helps to understand the monetary value of transactions, indicating how much customers typically spend in a single purchase. </p>",
            unsafe_allow_html=True
        )

    # Order Analysis Section
    st.header("Order Analysis")
    
    # Recency Distribution
    st.subheader("Recency Distribution")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_recency_distribution(df):
            df['recency'] = (df['ordered_date'].max() - df['ordered_date']).dt.days
            rfm = pd.DataFrame({'recency': df.groupby('ecom_reference_order_number')['recency'].min()})
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(rfm, x='recency', kde=True, ax=ax)
            ax.set_title('Recency Distribution')
            ax.set_xlabel('Days Since Last Purchase')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
        plot_recency_distribution(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Recency Distribution' plot is a histogram that illustrates the distribution of the number of days since the last recorded order. <br><br> This plot provides insight into order activity by showing how recently orders have been placed. The x-axis represents the number of days since the last order, while the y-axis shows the frequency of order IDs falling into each recency category. <br><br> Although this data does not directly track individual customers, it still offers valuable information about the frequency and timing of orders. </p>",
            unsafe_allow_html=True
        )


    # Order Quantity Statistics Section
    st.subheader("Order Quantity Statistics")
    # Create columns for layout
    col1, col2 = st.columns([1, 1])  # Adjust the column width ratios as needed
    with col1:
        # Function to display order quantity statistics
        def display_order_quantity_stats(df):
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
            st.table(stats_df)

        display_order_quantity_stats(data)
        with col2:
            st.markdown(
                "<p style='font-size:20px'> The 'Order Quantity Statistics' table presents summary statistics for the quantity of items ordered per transaction. <br><br> It provides key metrics such as the average, minimum, maximum, median, and quartiles (Q1 and Q3) of the total items ordered in each transaction. <br><br> This table offers a comprehensive overview of order sizes, highlighting typical order quantities, the range of quantities ordered, and the distribution's central tendency and spread.</p>",
                unsafe_allow_html=True
            )



    # Distribution of Items per Order
    st.subheader("Distribution of Items per Order")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_items_per_order_distribution(df):
            order_quantity_stats = df.groupby('ecom_reference_order_number')['ordered_quantity'].sum()
            fig, ax = plt.subplots(figsize=(6, 4))
            order_quantity_stats.plot(kind='hist', bins=30, alpha=0.7, ax=ax)
            ax.set_title('Distribution of Items per Order')
            ax.set_xlabel('Number of Items per Order')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
        plot_items_per_order_distribution(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Distribution of Items per Order' plot is a histogram that visualizes the frequency of different quantities of items ordered in a single transaction. <br><br> It shows how many transactions include specific numbers of items, providing insight into typical order sizes. <br><br> This plot helps identify common order sizes, whether customers often purchase single items or multiple items, and the overall distribution of items per transaction. </p>",
            unsafe_allow_html=True
        )
    
    # Pricing and Discount Analysis Section
    st.header("Pricing and Discount Analysis")

    # Distribution of Selling Prices and List Prices
    st.subheader("Distribution of Selling Prices and List Prices")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_price_distribution(df):
            fig, ax = plt.subplots(figsize=(6, 4))
            df['unit_selling_price'].plot(kind='hist', bins=20, alpha=0.5, label='Selling Price', ax=ax)
            df['unit_list_price'].plot(kind='hist', bins=20, alpha=0.5, label='List Price', ax=ax)
            ax.set_title('Distribution of Selling Prices and List Prices')
            ax.set_xlabel('Price')
            ax.set_ylabel('Frequency')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        plot_price_distribution(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Distribution of Selling Prices and List Prices' plot is a histogram that shows the frequency distribution of both the selling prices and list prices of products. <br><br> The plot helps to understand the range and distribution of prices at which products are sold (selling price) compared to their original or undiscounted prices (list price). </p>",
            unsafe_allow_html=True
        )

    # Distribution of Discount Percentages
    st.subheader("Distribution of Discount Percentages")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_discount_percentage_distribution(df):
            df['discount_percentage'] = (df['unit_list_price'] - df['unit_selling_price']) / df['unit_list_price'] * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            df['discount_percentage'].plot(kind='hist', bins=20, ax=ax)
            ax.set_title('Distribution of Discount Percentages')
            ax.set_xlabel('Discount Percentage')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
        plot_discount_percentage_distribution(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Distribution of Discount Percentages' plot is a histogram that displays the frequency distribution of discount percentages applied to products. <br><br> It provides insight into the range and prevalence of discounts offered on products. <br><br> This plot helps to identify common discount levels and the extent of price reductions across the product range. </p>",
            unsafe_allow_html=True
        )


    # Discount Percentage by Category
    st.subheader("Discount Percentage by Category")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_discount_percentage_by_category(df):
            df['discount_percentage'] = (df['unit_list_price'] - df['unit_selling_price']) / df['unit_list_price'] * 100
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='product_category', y='discount_percentage', data=df, width=0.5, ax=ax)
            ax.set_title('Discount Percentage by Category')
            ax.set_xlabel('Product Category')
            ax.set_ylabel('Discount Percentage')
            ax.set_xticklabels(df['product_category'].unique(), rotation=90, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        plot_discount_percentage_by_category(data_expanded)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Discount Percentage by Category' plot shows the variation in discount percentages across different product categories. <br><br> This visualization helps to understand how discounting strategies differ between categories and which categories offer higher or lower discounts. </p>",
            unsafe_allow_html=True
        )

    # Scatter Plot of Discount Percentage vs. Ordered Quantity
    st.subheader("Scatter Plot of Discount Percentage vs. Ordered Quantity")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_discount_vs_ordered_quantity(df):
            df['discount_percentage'] = (df['unit_list_price'] - df['unit_selling_price']) / df['unit_list_price'] * 100
            correlation = df['discount_percentage'].corr(df['ordered_quantity'])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='discount_percentage', y='ordered_quantity', data=df, alpha=0.6, ax=ax)
            ax.set_title('Scatter Plot of Discount Percentage vs. Ordered Quantity')
            ax.set_xlabel('Discount Percentage')
            ax.set_ylabel('Ordered Quantity')
            ax.grid(True)
            ax.text(0.05, max(df['ordered_quantity']) * 0.95, f'Correlation: {correlation:.2f}', fontsize=12, color='red')
            plt.tight_layout()
            st.pyplot(fig)
        plot_discount_vs_ordered_quantity(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Scatter Plot of Discount Percentage vs. Ordered Quantity' shows the relationship between the discount percentage applied to products and the quantity ordered. <br><br> It helps to analyze whether higher discounts correlate with higher sales volumes, providing insights into the effectiveness of discount strategies </p>",
            unsafe_allow_html=True
        )

    # Comparison of Average Discounts for Popular and Less Popular Items
    st.subheader("Comparison of Average Discounts for Popular and Less Popular Items")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_average_discounts_comparison_hist(df):
            item_popularity = df.groupby('product_name')['ordered_quantity'].sum()
            median_popularity = item_popularity.median()
            df['popularity'] = df['product_name'].map(lambda x: 'Popular' if item_popularity[x] > median_popularity else 'Less Popular')
            average_discounts = df.groupby('product_name')['discount_percentage'].mean()
            df['average_discount'] = df['product_name'].map(average_discounts)

            comparison_df = df[['product_name', 'popularity', 'average_discount']].drop_duplicates()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data=comparison_df, x='average_discount', hue='popularity', element='step', stat='density', common_norm=False, bins=20, ax=ax)
            ax.set_title('Comparison of Average Discounts for Popular and Less Popular Items')
            ax.set_xlabel('Average Discount Percentage')
            ax.set_ylabel('Density')
            plt.tight_layout()
            st.pyplot(fig)
        plot_average_discounts_comparison_hist(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Comparison of Average Discounts for Popular and Less Popular Items' plot is a histogram that compares the average discount percentages applied to popular and less popular items. <br><br> The items are categorized based on their sales volume, with 'popular' items having sales above the median and 'less popular' items below. <br><br> This plot helps to understand how discount strategies vary based on product popularity, showing the distribution of average discounts for each category. It provides insight into whether popular or less popular items receive larger discounts on average. </p>",
            unsafe_allow_html=True
        )

    # Box Plot of Unit Selling Prices by Top Performing Product Categories
    st.subheader("Box Plot of Unit Selling Prices by Top Performing Product Categories")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_top_categories_unit_selling_prices(df):
            df['total_sales_with_discount'] = df['ordered_quantity'] * df['unit_selling_price']
            category_sales = df.groupby('product_category')['total_sales_with_discount'].sum()
            top_categories = category_sales.sort_values(ascending=False).head(10).index.tolist()
            top_categories_df = df[df['product_category'].isin(top_categories)]

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='product_category', y='unit_selling_price', data=top_categories_df, order=top_categories, ax=ax)
            ax.set_title('Box Plot of Unit Selling Prices by Top Performing Product Categories')
            ax.set_xlabel('Product Category')
            ax.set_ylabel('Unit Selling Price')
            ax.set_xticklabels(top_categories, rotation=90)
            plt.tight_layout()
            st.pyplot(fig)
        plot_top_categories_unit_selling_prices(data_expanded)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Box Plot of Unit Selling Prices by Top Performing Product Categories' visualizes the range and distribution of unit selling prices across the top-performing product categories. <br><br> This plot provides insight into the pricing strategies within these categories, highlighting price variations and the median price point. </p>",
            unsafe_allow_html=True
        )
    
    # Supply Chain Analysis Section
    st.header("Supply Chain Analysis")

    # Top Distributing Companies by Quantity Shipped
    st.subheader("Top Distributing Companies by Quantity Shipped")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_top_distributors_treemap(df):
            top_distributors = df.groupby('operating_unit_name')['ordered_quantity'].sum().sort_values(ascending=False).head(10)
            colors = plt.cm.Blues(range(0, 256, int(256/len(top_distributors))))
            colors = colors[::-1]  # Reverse the colors
            fig, ax = plt.subplots(figsize=(6, 4))
            squarify.plot(sizes=top_distributors.values, label=top_distributors.index, alpha=0.8, color=colors, ax=ax)
            ax.set_title('Top Distributing Companies by Quantity Shipped')
            plt.axis('off')  # Remove axes
            plt.tight_layout()
            st.pyplot(fig)
        plot_top_distributors_treemap(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Top Distributing Companies by Quantity Shipped' treemap visualizes the quantity of items shipped by each distributor. It provides a clear view of the most active distribution companies, showing their relative contribution to the total shipments. <br><br> This visualization helps to understand the role and impact of different distributors in the supply chain. </p>",
            unsafe_allow_html=True
        )
    
    # Product Positioning Section
    st.header("Product Positioning")
    
    # Distribution of the Number of Items per Transaction
    st.subheader("Distribution of the Number of Items per Transaction")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_transaction_diversity(df):
            basket = (df
                    .groupby(['ecom_reference_order_number', 'product_name'])['product_name']
                    .count().unstack().reset_index().fillna(0)
                    .set_index('ecom_reference_order_number'))
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
            unique_items_per_transaction = basket_sets.sum(axis=1)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(unique_items_per_transaction, bins=range(1, unique_items_per_transaction.max() + 1), edgecolor='k')
            ax.set_title('Distribution of the Number of Items per Transaction')
            ax.set_xlabel('Number of Items per Transaction')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
        plot_transaction_diversity(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Transaction Diversity' plot shows the diversity in transactions by displaying the number of unique items purchased in each transaction. <br><br> It helps to understand whether customers are buying a wide variety of products or focusing on a few specific items. </p>",
            unsafe_allow_html=True
        )

    # Distribution of Item Frequencies
    st.subheader("Distribution of Item Frequencies")
    col1, col2 = st.columns([1, 1])
    with col1:
        def plot_item_frequencies(df):
            basket = (df
                    .groupby(['ecom_reference_order_number', 'product_name'])['product_name']
                    .count().unstack().reset_index().fillna(0)
                    .set_index('ecom_reference_order_number'))
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
            item_frequencies = basket_sets.sum(axis=0)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(item_frequencies, bins=range(1, item_frequencies.max() + 1), edgecolor='k')
            ax.set_title('Distribution of Item Frequencies')
            ax.set_xlabel('Frequency of Items')
            ax.set_ylabel('Number of Items')
            plt.tight_layout()
            st.pyplot(fig)
        plot_item_frequencies(data)
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Distribution of Item Frequencies' plot shows how frequently each product is purchased. <br><br> This plot provides insight into product popularity and helps identify which products are commonly bought together. </p>",
            unsafe_allow_html=True
        )

    # Transaction Diversity Statistics
    st.subheader("Transaction Diversity Statistics")
    col1, col2 = st.columns([1, 1])
    with col1:
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
    with col2:
        st.markdown(
            "<p style='font-size:20px'> The 'Transaction Diversity Statistics' table presents summary statistics regarding the number of unique items purchased per transaction. <br><br> It provides key metrics such as the mean, median, minimum, maximum, and quartiles (Q1 and Q3) of the unique items in each transaction. <br><br> This table offers an overview of how varied customer purchases are in terms of product diversity within a single transaction. </p>",
            unsafe_allow_html=True
        )

    # Item Frequencies Statistics
    st.subheader("Item Frequencies Statistics")
    col1, col2 = st.columns([1, 1])
    with col1:
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
    with col2:
        st.markdown(
            "<p style='font-size:20px'> The 'Item Frequency Statistics' table provides summary statistics on how frequently each item is purchased across all transactions. <br><br> It includes key metrics such as the mean, median, minimum, maximum, and quartiles (Q1 and Q3) for the frequency of item purchases. This table offers insights into the popularity of different products, highlighting how often items are bought on average, the range of purchase frequencies, and the distribution of these frequencies. </p>",
            unsafe_allow_html=True
        )

    # Considered as Niche Products
    st.subheader("Considered as Niche Products")
    col1, col2 = st.columns([1, 1])
    with col1:
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
    with col2:
        st.markdown(
            "<p style='font-size:20px'> <br> The 'Considered as Niche Products' section lists products identified as niche based on specific criteria. <br><br> These products are less frequently purchased but may cater to specific customer segments or interests. <br><br> Understanding niche products helps in targeting niche markets and optimizing inventory </p>",
            unsafe_allow_html=True
        )
        
        
    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        To Malia Group, By AUB Students
    </div>
    """, unsafe_allow_html=True)





elif page_selection == "Model Configuration":
    
        # Create three columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Leave the first column empty
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image(logo_resized, use_column_width=True)

    with col3:
        # Leave the third column empty
        st.write("")
        
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

        
    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        To Malia Group, By AUB Students
    </div>
    """, unsafe_allow_html=True)


elif page_selection == "Results and Visualization":
    
        # Create three columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Leave the first column empty
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image(logo_resized, use_column_width=True)

    with col3:
        # Leave the third column empty
        st.write("")
        
    # Results and Visualizations Page
    st.title("Results and Visualizations")

    # Load the generated rules
    if 'results' in st.session_state:
        results = st.session_state['results']

        # Scatter Plot of Support and Confidence
        st.subheader("Scatter Plot of Support and Confidence")
        col1, col2 = st.columns([2, 3])  # Adjust column width ratios as needed

        with col1:
            def plot_support_confidence(results):
                fig, ax = plt.subplots(figsize=(8, 5))  # Smaller figure size
                sns.scatterplot(data=results, x='support', y='confidence', size='lift', hue='lift', sizes=(20, 200), palette='viridis', ax=ax)
                ax.set_title('Support vs Confidence')
                ax.set_xlabel('Support')
                ax.set_ylabel('Confidence')
                ax.grid(True)
                st.pyplot(fig)

            plot_support_confidence(results)

        with col2:
            st.markdown(
                """
                <p style='font-size:28px'>
                <br> The 'Scatter Plot of Support and Confidence' visualizes the association rules generated from the dataset. 
                <br><br> Each point represents a rule, with its position determined by the rule's support (x-axis) and confidence (y-axis). 
                The size and color of the points are scaled by the rule's lift, highlighting stronger associations. 
                <br><br> This plot helps in identifying significant rules that may influence business decisions.
                </p>
                """, 
                unsafe_allow_html=True
            )

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
                    st.markdown(f"### Rule {index + 1}")
                    st.markdown(f"- **Antecedents:** {row['antecedents']}")
                    st.markdown(f"- **Consequents:** {row['consequents']}")
                    st.markdown(f"- **Support:** {row['support']:.3f}")
                    st.markdown(f"- **Confidence:** {row['confidence']:.3f}")
                    st.markdown(f"- **Lift:** {row['lift']:.3f}")
                    st.markdown(f"- **Leverage:** {row.get('leverage', 'N/A'):.3f}")
                    st.markdown(f"- **Conviction:** {row.get('conviction', 'N/A'):.3f}")
                    st.markdown(f"- **Zhang's Metric:** {row.get('zhangs_metric', 'N/A'):.3f}")
                    st.markdown("---")

        print_rules(filtered_results)

    else:
        st.write("Please run the model first to generate results.")
    
        
    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        To Malia Group, By AUB Students
    </div>
    """, unsafe_allow_html=True)


