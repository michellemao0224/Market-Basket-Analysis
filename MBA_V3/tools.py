import numpy as np

############   CUSTOMER   #############
def best_customer(df):
    # Get top ranked ranked customers based on the total amount
    customers_amounts = df.groupby('CustomerID')['TotalAmount'].agg(np.sum).sort_values(ascending=False)
    # Top 10 best customers
    customers_amounts.head(10).plot.bar()
    print(customers_amounts.head(10))

############   ITEM   #############
def popular_item(df):
    # Frequently sold popular items by quantitiy
    popular_items = df.groupby('Description')
    popular_items_quantitiy = popular_items['Quantity'].agg(np.sum).sort_values(ascending=False)
    # Top 10 popular items
    popular_items_quantitiy.head(10).plot.bar()
    print(popular_items_quantitiy.head(10))


############   DATE   #############
def sales_by_month(df):
    # Rank sales by month
    sales_month = df.sort_values('InvoiceDate').groupby(['InvoiceYear', 'InvoiceMonth'])
    sales_month_invoices = sales_month['InvoiceNo'].unique().agg(np.size)
    print(sales_month_invoices)
    sales_month_invoices.plot.bar()
    return sales_month

def sales_by_month_total(sales_month):
    # Rank sales by month total amounts
    sales_month_frq_amount= sales_month['TotalAmount'].agg(np.sum)
    # print(sales_month_frq_amount)
    sales_month_frq_amount.plot.bar()
    print(sales_month_frq_amount)

############   COUNTRY   #############
def sales_by_country(df):
    # Rank sales by countries
    sales_country = df.groupby('Country')
    # Order countries by total amount
    sales_country['TotalAmount'].agg(np.sum).sort_values(ascending=False)
    # Order countries by number of invoices
    sales_country['InvoiceNo'].unique().agg(np.size).sort_values(ascending=False)
    # Order countries by number of customers
    sales_country['CustomerID'].unique().agg(np.size).sort_values(ascending=False)

############ FILTER #############
def search_item(df):
    df['Description'] = df['Description'].astype('str')
    # Search for particular item
    df = df[df['Description'].str.contains('FASHION')]
    return df