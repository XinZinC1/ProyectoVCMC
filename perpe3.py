import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
tamaño_img=720

# Cargar los datos
with open('x.pickle', 'rb') as f:
    x = pickle.load(f)

with open('y.pickle', 'rb') as f:
    y = pickle.load(f)

# Convertir los datos a formato NumPy
x = np.array(x)
y = np.array(y)

# Normalizar los datos
x = x / 255.0

# Convertir las etiquetas a categorías
y = to_categorical(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crear la red neuronal
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(tamaño_img, tamaño_img, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Compilar la red neuronal
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar los callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

# Entrenar la red neuronal
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint])