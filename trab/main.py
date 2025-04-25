# kmeans_diabetes_cluster.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregamento e pré-processamento dos dados
data = pd.read_csv('C:/Users/EDUARDOFREITASGRUNIT/Downloads/diabetes 130-us hospitals for years 1999-2008/diabetic_data.csv')

# Remover colunas de identificadores irrelevantes para clustering
cols_to_drop = ['encounter_id', 'patient_nbr']
data.drop(columns=cols_to_drop, inplace=True)

# Remover colunas com alta cardinalidade irrelevante ou muitos NA
data.replace('?', np.nan, inplace=True)
data.dropna(axis=1, thresh=0.8 * len(data), inplace=True)

# Preencher valores ausentes com 'missing' em categóricas
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    data[col] = data[col].fillna('missing')

# Codificar variáveis categóricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Normalização
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 2. Determinação do número ótimo de clusters
inertias = []
silhouettes = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(data_scaled, kmeans.labels_))

# Plotar curvas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, marker='o', color='green')
plt.title('Score de Silhueta')
plt.xlabel('Número de clusters')
plt.ylabel('Silhueta')

plt.tight_layout()
plt.show()

# 3. Treinamento final do modelo com k=4 (exemplo, ajustar com base no gráfico)
k = 4
kmeans_final = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_final.fit(data_scaled)

print("\nCentroides:")
print(kmeans_final.cluster_centers_)

# 4. Módulo de inferência
def inferir_cluster(nova_instancia: pd.DataFrame) -> int:
    """Recebe um DataFrame com 1 linha, retorna cluster e exibe características."""
    for col in nova_instancia.columns:
        if col in label_encoders:
            nova_instancia[col] = label_encoders[col].transform(nova_instancia[col].fillna('missing'))

    nova_instancia_scaled = scaler.transform(nova_instancia)
    cluster = kmeans_final.predict(nova_instancia_scaled)[0]
    print(f"\nA nova instância pertence ao cluster: {cluster}")
    return cluster

# Exemplo de uso:
# nova = data.sample(1).drop(columns=['readmitted'])
# inferir_cluster(nova)
