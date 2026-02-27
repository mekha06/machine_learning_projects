import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

def build_rfm(df):

    # -----------------------------
    # Date conversion (fixed)
    # -----------------------------
    df['InvoiceDate'] = pd.to_datetime(
        df['InvoiceDate'],
        dayfirst=True,
        errors='coerce'
    )

    df = df.dropna(subset=['InvoiceDate'])
    snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # -----------------------------
    # Build RFM
    # -----------------------------
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency','Frequency','Monetary']
    rfm.reset_index(inplace=True)

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[['Recency','Frequency','Monetary']]
    )

    # -----------------------------
    # KMeans Clustering
    # -----------------------------
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------
    sil_score = silhouette_score(rfm_scaled, rfm['Cluster'])
    db_score  = davies_bouldin_score(rfm_scaled, rfm['Cluster'])
    ch_score  = calinski_harabasz_score(rfm_scaled, rfm['Cluster'])

    return rfm, sil_score, db_score, ch_score