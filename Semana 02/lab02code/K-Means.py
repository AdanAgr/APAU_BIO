import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Cargar el dataset
file_path = "./Mall_Customers.csv"  
df = pd.read_csv(file_path)

# Eliminar la columna ID
df_clustering = df.iloc[:, 1:]  # Ahora incluye Gender

# Convertir Gender de texto a numérico usando un bucle
for i in range(len(df_clustering)):
    if df_clustering.loc[i, "Gender"] == "Female":
        df_clustering.loc[i, "Gender"] = 0
    elif df_clustering.loc[i, "Gender"] == "Male":
        df_clustering.loc[i, "Gender"] = 1

# Convertir la columna Gender a tipo numérico (int)
df_clustering["Gender"] = df_clustering["Gender"].astype(int)

# Renombrar las columnas
df_clustering.columns = ["Gender", "Age", "Annual_Income", "Spending_Score"]

# Aplicar K-means
k = 5  # Número de clústeres a elegir
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(df_clustering)

# Agregar etiquetas de clúster al dataset
df_clustering["Cluster"] = kmeans.labels_

# Visualización en 2D (usando dos características principales)
plt.figure(figsize=(8, 6))
plt.scatter(df_clustering["Annual_Income"], df_clustering["Spending_Score"], c=df_clustering["Cluster"], cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s=300, c='red', marker='X', label='Centroids')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Clustering con K-Means")
plt.legend()
plt.show()

# Coeficiente de Silhouette
silhouette_avg = silhouette_score(df_clustering, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

# Índice de Davies-Bouldin
db_score = davies_bouldin_score(df_clustering, kmeans.labels_)
print("Davies-Bouldin Score:", db_score)

# Índice de Calinski-Harabasz
ch_score = calinski_harabasz_score(df_clustering, kmeans.labels_)
print("Calinski-Harabasz Score:", ch_score)