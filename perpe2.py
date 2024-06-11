import pickle
import matplotlib.pyplot as plt

# Cargar el archivo pickle
with open('y.pickle', 'rb') as f:
    y = pickle.load(f)

# Mostrar los datos
print(y)
