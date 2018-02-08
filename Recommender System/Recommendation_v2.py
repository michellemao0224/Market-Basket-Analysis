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
# print(itemTable)

# print(itemTable.index.get_level_values(0))
product = []
for name in itemTable.index:
    product.append(name)
    # print(name)
    # print(itemTable.loc[name])
    # df['Product'] = itemTable.loc[name]
df['Product'] = pd.Series(product)
# print(df['Product'])

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
# print(tfidf_matrix.shape)
# (531285, 2129)
# over 531285 different words were used to describe the 2129 products in the dataset

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and product descriptions
indices = pd.Series(df.index, index=df['Product']).drop_duplicates()

# Function that takes in product description as input and outputs most similar products
def get_recommendations(product, cosine_sim=cosine_sim):
    # Get the index of the product that matches the description
    idx = indices[product]

    # Get the pairwsie similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar products
    sim_scores = sim_scores[1:11]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar products
    return df['Product'].iloc[product_indices]

# ############ Change your product name here ##############
# # print(get_recommendations('ALARM CLOCK BAKELIKE PINK'))
# result = get_recommendations('ALARM CLOCK BAKELIKE PINK')
#
# # Write MBA Report
# my_df = pd.DataFrame(result)
# my_df.to_csv('recomend_product.csv', index=False, header=True)
# print(my_df)

# print(get_recommendations('BOX OF 6 ASSORTED COLOUR TEASPOONS'))


df = df[(df['CustomerID'] == 17850)]
print(df)

itemTable = df.groupby(['Description']).agg({
    'Quantity': lambda x: x.sum(),
    'TotalPrice': lambda x: x.sum()}).sort_values(by = 'TotalPrice', ascending=False)

product = []
for name in itemTable.index:
    product.append(name)
    # print(name)
    BuyingHistory = pd.Series(product[0:3])
print(BuyingHistory)

######################  ISSUE !!!! ######################
for item in BuyingHistory:
    result = get_recommendations(item)
######################  ISSUE !!!! ######################

# Write MBA Report
my_df = pd.DataFrame(result)
my_df.to_csv('recomend_product.csv', index=False, header=True)
print(my_df)