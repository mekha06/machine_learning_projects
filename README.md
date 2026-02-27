# 🧠 AI Customer Segmentation & Product Recommendation System

An end-to-end machine learning application that performs **customer segmentation using RFM analysis** and generates **personalized product recommendations**.
Built with Python, Scikit-Learn, and Streamlit.

---

## 📌 Project Overview

Businesses need to understand customer behavior to design targeted marketing strategies.
This project segments customers into meaningful groups and recommends products based on purchase patterns.

Key capabilities:

* 📊 Customer segmentation using K-Means clustering
* 🧮 RFM (Recency, Frequency, Monetary) analysis
* 🎯 Personalized product recommendations
* 📉 Cluster evaluation metrics
* 🖥️ Interactive Streamlit dashboard

---

## 📂 Dataset

Dataset used: **Online Retail Dataset** from the UCI Machine Learning Repository

**Features include:**

* InvoiceNo — Transaction ID
* StockCode — Product ID
* Description — Product name
* Quantity — Units purchased
* InvoiceDate — Date & time of transaction
* UnitPrice — Price per unit
* CustomerID — Unique customer ID
* Country — Customer location

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing

* Removed cancelled transactions
* Removed negative or zero quantities/prices
* Handled missing Customer IDs
* Created TotalPrice = Quantity × UnitPrice
* Converted date formats

---

### 2️⃣ RFM Feature Engineering

For each customer:

* **Recency** → Days since last purchase
* **Frequency** → Number of transactions
* **Monetary** → Total spending

---

### 3️⃣ Customer Segmentation

Algorithm: K-Means Clustering

Features used:

* Recency
* Frequency
* Monetary

Data scaled using StandardScaler.

---

### 4️⃣ Cluster Evaluation Metrics

To validate clustering quality, three metrics are used:

| Metric                  | Interpretation     | Ideal Value |
| ----------------------- | ------------------ | ----------- |
| Silhouette Score        | Cluster separation | Higher      |
| Davies–Bouldin Index    | Cluster similarity | Lower       |
| Calinski–Harabasz Score | Dispersion ratio   | Higher      |

---

### 5️⃣ Product Recommendation

Recommends products to a selected customer based on:

* Purchase behavior
* Similar customers
* Product popularity within cluster

---

## 🖥️ Streamlit Application Features

* Select customer ID
* View customer profile (RFM values + cluster)
* Visualize customer segments
* Display evaluation metrics
* Show top recommended products

---

## 📁 Project Structure

```
cust_segmentation/
│
├── app.py                 # Streamlit UI
├── online_retail.csv      # Dataset
├── requirements.txt
│
└── src/
    ├── preprocessing.py
    ├── segmentation.py
    ├── recommender.py
    └── evaluation.py
```

---

## ▶️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone <your-repo-link>
cd cust_segmentation
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8502
```

---

## 📊 Example Output

* Customer profile with RFM values
* Assigned cluster
* Cluster quality metrics
* Recommended products ranked by relevance

---

## 🧠 Use Cases

* Targeted marketing campaigns
* Customer lifetime value analysis
* Cross-selling & upselling
* Personalized e-commerce recommendations
* Business intelligence dashboards

---

## 🛠️ Tech Stack

* Python
* Pandas & NumPy
* Scikit-Learn
* Seaborn & Matplotlib
* Streamlit

---

## 🚀 Future Improvements

* Hybrid recommendation models
* Deep learning–based recommendations
* Real-time data integration
* Deployment on cloud platforms
* Explainable AI insights

---

