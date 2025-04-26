import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('future.no_silent_downcasting', True)

data = pd.read_csv('C:/Users/EDUARDOFREITASGRUNIT/Downloads/diabetes 130-us hospitals for years 1999-2008/diabetic_data.csv')
data = data.iloc[:1167]  # reduz para facilitar testes e performance

data.drop(columns=['encounter_id', 'patient_nbr'], inplace=True)
data.replace('?', np.nan, inplace=True)

data.dropna(axis=1, thresh=0.8 * len(data), inplace=True)

possiveis_features = [
    'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency', 
    'number_inpatient', 'number_diagnoses', 'weight', 'payer_code',
    'medical_specialty', 'max_glu_serum', 'A1Cresult'
]

features = [col for col in possiveis_features if col in data.columns]

categorical_cols = [col for col in ['age', 'weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult'] if col in data.columns]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].fillna('missing'))
    label_encoders[col] = le

scaler = StandardScaler()
data_model = data[features].copy()
data_scaled = scaler.fit_transform(data_model)

inertias = []
silhouettes = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(data_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, marker='o', color='blue')
plt.title('Coeficiente de Silhueta')
plt.xlabel('Número de clusters')
plt.ylabel('Silhueta')

plt.tight_layout()
plt.show()

k = 4
kmeans_final = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_final.fit(data_scaled)

print("\nCentroides (valores normalizados):")
print(pd.DataFrame(kmeans_final.cluster_centers_, columns=features))

data['cluster'] = kmeans_final.labels_

print("\nMédia das variáveis por cluster:")
print(data.groupby('cluster')[features].mean())

sns.countplot(x='cluster', data=data)
plt.title('Distribuição dos Pacientes por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Contagem')
plt.show()

def inferir_cluster(nova_instancia: pd.DataFrame) -> int:
    """Recebe uma nova instância, informa o cluster e características."""
    nova_instancia = nova_instancia[features].copy()

    for col in categorical_cols:
        nova_instancia[col] = nova_instancia[col].fillna('missing')
        try:
            nova_instancia[col] = label_encoders[col].transform(nova_instancia[col])
        except ValueError:
            nova_instancia[col] = nova_instancia[col].apply(
                lambda x: -1 if x not in label_encoders[col].classes_ else label_encoders[col].transform([x])[0]
            )

    nova_scaled = scaler.transform(nova_instancia)
    cluster = kmeans_final.predict(nova_scaled)[0]
    print(f"\nA nova instância pertence ao cluster: {cluster}")
    print("Características da nova instância:")
    print(nova_instancia.to_string(index=False))
    return cluster

if __name__ == "__main__":
    nova = data.sample(1).drop(columns=['readmitted', 'cluster'], errors='ignore')
    inferir_cluster(nova)
