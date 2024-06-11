import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
from sklearn.preprocessing import StandardScaler

# Cargar los datos de entrada y salida desde archivos pickle
with open('x.pickle', 'rb') as f:
    X = pickle.load(f)

with open('y.pickle', 'rb') as f:
    Y = pickle.load(f)

# Normalizar los datos de X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir la estructura de la red neuronal
n_capas = 3
neuronas_capa_anterior = [X_scaled.shape[1], 4, 1]
funciones_activacion = [tf.nn.relu, tf.nn.relu, tf.nn.sigmoid]

# Crear la red neuronal
red_neuronal = []
for paso in range(n_capas - 1):
    neurona = Dense(
        neuronas_capa_anterior[paso + 1],
        activation=funciones_activacion[paso],
        input_shape=(neuronas_capa_anterior[paso],)
    )
    red_neuronal.append(neurona)

# Crear la capa de salida
neurona_salida = Dense(1, activation='sigmoid')

# Crear el modelo de la red neuronal
modelo = Sequential(red_neuronal + [neurona_salida])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar la red neuronal
modelo.fit(X_scaled, Y, epochs=1000, batch_size=4, verbose=2)

# Evaluar la red neuronal
Y_pred = modelo.predict(X_scaled)
print(Y_pred)