import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

data_file = r"C:\FINALS\data\Mall_Customers.csv"

if not os.path.isfile(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}")

df = pd.read_csv(data_file)
print("Data preview (first 5 rows):")
print(df.head())

features = ['Annual Income (k$)', 'Spending Score (1-100)']
if not all(col in df.columns for col in features):
    raise KeyError(f"Expected columns not found. Available columns: {df.columns.tolist()}")

X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertias, marker='o')
plt.title("Elbow Method: Inertia vs. k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.grid(True)
plt.show()
best_k = 5  # you may adjust after inspecting the elbow plot
print(f"\nChoosing k = {best_k}")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

sil_score = silhouette_score(X_scaled, labels) if best_k > 1 else None
print(f"Silhouette Score: {sil_score:.3f}" if sil_score is not None else "Silhouette Score not defined for k=1")

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=features[0],
    y=features[1],
    hue='Cluster',
    data=df,
    palette='Set1',
    s=80
)
plt.title(f"Customer Segments (K-Means, k={best_k})")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.legend(title="Cluster")
plt.show()

centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=features)
print("\nCluster centers (approximate):")
print(centers_df)

print("\nInterpretation of clusters:")
print("""
- Cluster 0: low income, low spending → Conservative buyers.
- Cluster 1: high income, high spending → Premium customers.
- Cluster 2: low income, high spending → Impulsive buyers.
- Cluster 3: average income and spending → Middle segment.
- Cluster 4: high income, low spending → Cautious rich customers.
""")

out_file = r"C:\FINALS\clustered_customers.csv"
df.to_csv(out_file, index=False)
print(f"\nSaved clustered dataset to: {out_file}")
