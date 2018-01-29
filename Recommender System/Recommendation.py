##########  Recommendation System ##########

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df = pd.read_excel("Online_Retail.xlsx")
df.head()

df = df[(df['Quantity']>0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

itemTable = df.groupby(['Description']).agg({
    'Quantity': lambda x: x.sum(),
    'TotalPrice': lambda x: x.sum()})
# print(itemTable.head(10))

# print(itemTable.index.get_level_values(0))
product = []
for name in itemTable.index:
    product.append(name)
    # print(name)
    # print(itemTable.loc[name])
    # df['Product'] = itemTable.loc[name]
df['Product'] = pd.Series(product)
print(df['Product'])

#  Content Based Recommender
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['Product'] = df['Product'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['Product'])

#Output the shape of tfidf_matrix
print(tfidf_matrix.shape)
# over 75,000 different words were used to describe the 45,000 movies in your dataset


# print(df[df['CustomerID']== 17850])

descrptions = df[df['CustomerID']== 17850]['Description']
