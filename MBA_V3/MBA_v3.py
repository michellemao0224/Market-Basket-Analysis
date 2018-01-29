from data import open_file, add_fields, data_prepare, write_csv
from tools import best_customer, popular_item, sales_by_country, sales_by_month, sales_by_month_total, search_item
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# -------------------------------------------------------------------

# Data Reader
path = "Online_Retail.xlsx"
df = open_file(path)

# Check Country info
# print(df.Country.value_counts().reset_index().head(n = 10))
# Check Quantity info
# print(df.Quantity.describe())
# Check UnitPrice info
# print(df.UnitPrice.describe())

# Add Total Amount & Invoice Date fields
df = add_fields(df)
# Data Preprocessing
df = data_prepare(df)
# Filter: search for particular item
df = search_item(df)

# -------------------------------------------------------------------

# Prepare for Market Basket Analysis
# basket for the whole sales
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# basket for different country
# basket = (df[df['Country'] =="France"]
#           .groupby(['InvoiceNo', 'Description'])['Quantity']
#           .sum().unstack().reset_index().fillna(0)
#           .set_index('InvoiceNo'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

# -------------------------------------------------------------------

# Frequent Items
# Need to adjust minimum support to find the best result
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
# print(rules.head())

# -------------------------------------------------------------------

# Association Rules
# If we are ony interested in rules that satisfy the following criteria:
#
#   1.  at least 2 antecedants
#   2.  a confidence > 0.7
#   3.  a lift score > 6

rules['antecedant_len'] = rules['antecedants'].apply(lambda x: len(x))
rules

# Need to adjust the rules' value to find the best result
# result = rules[ (rules['antecedant_len'] >= 2) &
#        (rules['confidence'] > 0.7) &
#        (rules['lift'] > 6)]

result = rules[ (rules['confidence'] > 0.7) &
                (rules['lift'] > 1)]

# -------------------------------------------------------------------

# Write MBA Report
my_df = write_csv(result)