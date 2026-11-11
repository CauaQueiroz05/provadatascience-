import pandas as pd
import numpy as np
import joblib

num_cols = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']
cat_cols = ['chas']

def inferir(dados_nova_casa):
    mod = joblib.load('models/kmeans_model.pkl')
    sc = joblib.load('models/scaler.pkl')
    
    nova = pd.DataFrame([dados_nova_casa])
    nova_scaled = pd.DataFrame(sc.transform(nova[num_cols]), columns=num_cols)
    X_nova = pd.concat([nova_scaled, nova[cat_cols]], axis=1)[num_cols + cat_cols]
    
    grupo = mod.predict(X_nova)[0]
    centroide = mod.cluster_centers_[grupo]
    
    centroide_num = sc.inverse_transform(centroide[:len(num_cols)].reshape(1, -1))[0]
    
    resultado = dict(zip(num_cols, np.round(centroide_num, 2)))
    resultado['chas'] = round(centroide[len(num_cols)], 2)
    
    print("Grupo:", grupo)
    print("Centroide desnormalizado:")
    print(resultado)

nova_casa = {
    'crim': 0.03, 'zn': 0.0, 'indus': 7.07, 'chas': 0, 'nox': 0.469, 
    'rm': 6.421, 'age': 78.9, 'dis': 4.9671, 'rad': 2, 'tax': 242, 
    'ptratio': 17.8, 'b': 396.9, 'lstat': 9.14, 'medv': 21.6
}

inferir(nova_casa)