import pandas as pd

def load_and_clean_data(path):

    df = pd.read_csv(path, encoding="ISO-8859-1")
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(str)

    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    return df.reset_index(drop=True)