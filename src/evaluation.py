from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def clustering_metrics(rfm_scaled, labels):

    db = davies_bouldin_score(rfm_scaled, labels)
    ch = calinski_harabasz_score(rfm_scaled, labels)
    print(f"Davies-bouldin-score:{db:.3f}")

    return db, ch
