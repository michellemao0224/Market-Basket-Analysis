import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np

# Data Reader
df = pd.read_excel("Online_Retail.xlsx")
# print(df.dtypes)
df.head()

# Check Country info
# print(df.Country.value_counts().reset_index().head(n = 10))
# Check Quantity info
# print(df.Quantity.describe())
# Check UnitPrice info
# print(df.UnitPrice.describe())

# Add Total Amount & Invoice Date fields
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
df['InvoiceYear'] = df['InvoiceDate'].dt.year
df['InvoiceMonth'] = df['InvoiceDate'].dt.month
df['InvoiceYearMonth'] = df['InvoiceYear'].map(str) + "-" + df['InvoiceMonth'].map(str)
print(df.describe())

############   CUSTOMER   #############
# Get top ranked ranked customers based on the total amount
customers_amounts = df.groupby('CustomerID')['TotalAmount'].agg(np.sum).sort_values(ascending=False)
# Top 10 best customers
customers_amounts.head(10).plot.bar()

############   ITEM   #############
# Frequently sold popular items by quantitiy
popular_items = df.groupby('Description')
popular_items_quantitiy = popular_items['Quantity'].agg(np.sum).sort_values(ascending=False)
# Top 10 popular items
popular_items_quantitiy.head(10).plot.bar()

############   DATE   #############
# Rank sales by month
sales_month = df.sort_values('InvoiceDate').groupby(['InvoiceYear', 'InvoiceMonth'])
sales_month_invoices = sales_month['InvoiceNo'].unique().agg(np.size)
# print(sales_month_invoices)
sales_month_invoices.plot.bar()

# Rank sales by month total amounts
sales_month_frq_amount= sales_month['TotalAmount'].agg(np.sum)
# print(sales_month_frq_amount)
sales_month_frq_amount.plot.bar()

############   COUNTRY   #############
# Rank sales by countries
sales_country = df.groupby('Country')
# Order countries by total amount
sales_country['TotalAmount'].agg(np.sum).sort_values(ascending=False)
# Order countries by number of invoices
sales_country['InvoiceNo'].unique().agg(np.size).sort_values(ascending=False)
# Order countries by number of customers
sales_country['CustomerID'].unique().agg(np.size).sort_values(ascending=False)

## Data Preprocessing
# Remove empty space in Description
df['Description'] = df['Description'].str.strip()
# Drop missing values in InvoiceNo
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
# Remove credit card transactions
df = df[~df['InvoiceNo'].str.contains('C')]

# Prepare for market basket
# basket for the whole sales
# basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
#           .sum().unstack().reset_index().fillna(0)
#           .set_index('InvoiceNo'))

# basket for different country
basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)


# Frequent Items
# Need to adjust minimum support to find the best result
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules.head())

# Association Rules
# If we are ony interested in rules that satisfy the following criteria:
#
#   1.  at least 2 antecedants
#   2.  a confidence > 0.8
#   3.  a lift score > 6

rules["antecedant_len"] = rules["antecedants"].apply(lambda x: len(x))
rules

# Need to adjust the rules' value to find the best result
result = rules[ (rules['antecedant_len'] >= 2) &
       (rules['confidence'] > 0.7) &
       (rules['lift'] > 6)]

# Write MBA Report
my_df = pd.DataFrame(result)
my_df.to_csv('report.csv', index=False, header=True)
print(my_df)






