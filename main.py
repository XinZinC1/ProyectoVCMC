import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
DATADIR=r"ClasificacionHojas"
CATEGORIAS=["ENFERMA","SANA"]
def Generar_datos():
    for categoria in CATEGORIAS:
        carpeta= os.path.join(DATADIR,categoria)
        # Parámetros de edición
        brillo = 1.75  # Ajusta el brillo (1.0 = sin cambios)
        contraste = 1.05  # Ajusta el contraste (1.0 = sin cambios)
        altas_luces = 1.05  # Ajusta las altas luces (1.0 = sin cambios)
        sombras = 1.05 # Ajusta las sombras (1.0 = sin cambios)
        nitidez = 1.5  # Ajusta la nitidez (1.0 = sin cambios)

        for filename in os.listdir(carpeta):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                ruta = os.path.join(carpeta, filename)
                imagen = Image.open(ruta)

                # Ajusta el brillo
                ajuste_brillo = ImageEnhance.Brightness(imagen)
                imagen = ajuste_brillo.enhance(brillo)
                # Ajusta el contraste
                ajuste_contraste = ImageEnhance.Contrast(imagen)
                imagen = ajuste_contraste.enhance(contraste)
                # Ajusta las altas luces
                ajuste_altasluces = ImageEnhance.Brightness(imagen)
                imagen = ajuste_altasluces.enhance(altas_luces)
                # Ajusta las sombras
                ajuste_sombra = ImageEnhance.Brightness(imagen)
                imagen = ajuste_sombra.enhance(sombras)
                # Ajusta la nitidez
                ajuste_nitidez = ImageEnhance.Sharpness(imagen)
                imagen = ajuste_nitidez.enhance(nitidez)

                # Convertir la imagen a formato OpenCV
                imagen_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
                # Aplicar filtro de Sobel
                bordes_verticales = cv2.Sobel(imagen_cv, cv2.CV_8U, 1, 0, ksize=3)
                bordes_horizontales = cv2.Sobel(imagen_cv, cv2.CV_8U, 0, 1, ksize=3)
                gradiente = cv2.addWeighted(bordes_verticales, 0.5, bordes_horizontales, 0.5, 0)

                # Mostrar la imagen con el filtro de Sobel
                plt.imshow(gradiente, cmap='gray')
                plt.show()

Generar_datos()