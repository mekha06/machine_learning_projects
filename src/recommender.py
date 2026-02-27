import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_recommender(df):

    user_item = df.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )

    sim = cosine_similarity(user_item.T)

    sim_df = pd.DataFrame(
        sim,
        index=user_item.columns,
        columns=user_item.columns
    )

    return user_item, sim_df


def recommend(customer_id, user_item, sim_df, n=10):

    if customer_id not in user_item.index:
        return None

    purchased = user_item.loc[customer_id]
    purchased = purchased[purchased > 0].index.tolist()

    scores = pd.Series(dtype=float)

    for item in purchased:
        scores = scores.add(sim_df[item], fill_value=0)

    scores = scores.drop(purchased)
    return scores.sort_values(ascending=False).head(n)