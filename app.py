import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from src.preprocessing import load_and_clean_data
from src.segmentation import build_rfm
from src.recommender import build_recommender, recommend

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_and_clean_data("data/Online Retail.xlsx")

rfm, sil_score, db_score, ch_score = build_rfm(df)

user_item, sim_df = build_recommender(df)

# --------------------------------------------------
# UI
# --------------------------------------------------

st.set_page_config(layout="wide")

st.title("🧠 Customer Segmentation & Recommendation System")

# Sidebar
customer = st.sidebar.selectbox("Select Customer ID", rfm['CustomerID'])

# --------------------------------------------------
# CUSTOMER PROFILE
# --------------------------------------------------

st.header("📊 Customer Profile")

cust_data = rfm[rfm['CustomerID'] == customer]

st.write(cust_data)

st.metric("Cluster", int(cust_data['Cluster']))
st.subheader("📊 Clustering Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Silhouette Score (Higher is Better)", f"{sil_score:.3f}")
col2.metric("Davies–Bouldin Index (Lower is Better)", f"{db_score:.3f}")
col3.metric("Calinski–Harabasz Score (Higher is Better)", f"{ch_score:.1f}")


# --------------------------------------------------
# CLUSTER VISUALIZATION
# --------------------------------------------------

st.header("🔍 Customer Segments")

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=rfm,
    x='Frequency',
    y='Monetary',
    hue='Cluster',
    palette='Set2',
    ax=ax
)

st.pyplot(fig)

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

st.header("🛍️ Recommended Products")

recs = recommend(customer, user_item, sim_df)

if recs is not None:
    st.dataframe(recs.reset_index().rename(
        columns={'index':'Product','0':'Score'}))
else:
    st.write("Customer not found.")
