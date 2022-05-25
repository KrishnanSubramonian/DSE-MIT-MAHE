import streamlit as st
import pandas as pd

import gzip
import simplejson

def parse(filename):
    f = open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos+2:]
        entry[eName] = rest
  yield entry

# for e in parse("Watches.txt"):
#   print(simplejson.dumps(e))

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


df_watch = getDF('Watches.txt')

from surprise import Reader, Dataset, SVD
from surprise.model_selection.validation import cross_validate


def get_recom_watches(Input):
    from surprise import Reader, Dataset, SVD
    from surprise.model_selection.validation import cross_validate

    reader = Reader()

    data1 = Dataset.load_from_df(df_watch[['review/userId', 'product/productId', 'review/score']], reader)

    svd1 = SVD()

    cross_validate(svd1, data1, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    trainset1 = data1.build_full_trainset()
    svd1.fit(trainset1)

    # Recommending Products:

    titles1 = df_watch[['product/productId', 'product/title']].copy()
    titles1.head()

    titles1['Estimate_Score'] = titles1['product/productId'].apply(lambda x: svd1.predict(Input, x).est)

    titles1 = titles1.sort_values(by=['Estimate_Score'], ascending=False)

    data_t_1 = titles1.copy()

    data_t_1.drop_duplicates(subset="product/productId", keep=False, inplace=True)

    return data_t_1.head(5)

get_recom_watches('AEM9CCSE7CQ9M')


















'''

header = st.container()

with header:
    st.title("AMAZON DATA SET PREDICTIONS: ")

take = st.container()

input_id = take.text_input("Enter the User ID: ", 'AEM9CCSE7CQ9M')

recomendations = st.container()

recomendations.subheader('The Recomendations are: ')

for i in range(1, 6):
    reco = st.container()
    reco.write(data_t_1.iloc[[i], [1]]['product/title'])

'''

