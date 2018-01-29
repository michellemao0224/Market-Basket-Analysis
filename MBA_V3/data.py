import pandas as pd

# Open and read an Excel file
def open_file(path):
    df = pd.read_excel(path)
    # print(df.dtypes)
    # print(df.head())
    return df

# Add Total Amount & Invoice Date fields
def add_fields(df):
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceYear'] = df['InvoiceDate'].dt.year
    df['InvoiceMonth'] = df['InvoiceDate'].dt.month
    df['InvoiceYearMonth'] = df['InvoiceYear'].map(str) + "-" + df['InvoiceMonth'].map(str)
    # print(df.describe())
    return df

# Data Preprocessing
def data_prepare(df):
    # Drop missing values in InvoiceNo
    df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
    df['InvoiceNo'] = df['InvoiceNo'].astype('str')
    # Remove credit card transactions
    df = df[~df['InvoiceNo'].str.contains('C')]
    return df

# Write MBA Report
def write_csv(result):
    my_df = pd.DataFrame(result)
    my_df.to_csv('report.csv', index=False, header=True)
    print(my_df)
