import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

num_cols = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']
cat_cols = ['chas']

scaler = StandardScaler()
df_scaled_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
X = pd.concat([df_scaled_num, df[cat_cols].reset_index(drop=True)], axis=1)

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('K')
plt.ylabel('Inercia')
plt.show()

k_otimo = 3
modelo = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
modelo.fit(X)

os.makedirs('models', exist_ok=True)
joblib.dump(modelo, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Treinamento concluido e modelos salvos!")