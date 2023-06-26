import numpy as np
import cv2
from skimage.filters import gaussian
import math
from PIL import Image


########################### UNSHARP MASKING ###############################

class unsharp_mask_class():

    def __init__(self):
        pass

    def unsharp(self,r_image,filtered_image,scale_factor):
        #filtered_image = gaussian(filtered_image,sigma=1,mode='constant',cval=0.0)
        img = scale_factor*(r_image - filtered_image)
        unsharp_mask = r_image + img
       
        return unsharp_mask


######################## HISTOGRAM TECHNIQUES #############################

class histog():

    def __init__(self):
        pass
    
    def histogram_equalization(self, image):
        # Obtener el histograma de la imagen
        hist = np.zeros(256, dtype=int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i, j]] += 1

        # Calcular la función de distribución acumulativa (CDF)
        cdf = np.cumsum(hist)

        # Normalizar el CDF al rango [0, 1]
        cdf_normalized = cdf / float(np.sum(hist))

        # Aplicar la función exponencial para ajustar el histograma
        gamma = 0.5  # Parámetro de ajuste
        adjusted_cdf = cdf_normalized ** gamma

        # Mapear los valores de píxeles originales a los nuevos valores ajustados
        adjusted_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel_value = image[i, j]
                adjusted_pixel_value = int(adjusted_cdf[pixel_value] * 255)
                adjusted_image[i, j] = adjusted_pixel_value

        return adjusted_image
    

    def sqrt_contrast(self, image):
        # Obtener las dimensiones de la imagen
        height, width = image.shape

        # Aplicar la función de raíz cuadrada a cada píxel de la imagen
        adjusted_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                pixel_val = image[i, j]
                adjusted_val = int(np.sqrt(pixel_val))
                adjusted_image[i, j] = adjusted_val

        return adjusted_image


    def linear_contrast(self, image):
        # Obtener las dimensiones de la imagen
        height, width = image.shape

        # Encontrar el valor mínimo y máximo de intensidad en la imagen
        min_val = float('inf')
        max_val = float('-inf')
        for i in range(height):
            for j in range(width):
                pixel_val = image[i, j]
                if pixel_val < min_val:
                    min_val = pixel_val
                if pixel_val > max_val:
                    max_val = pixel_val

        # Calcular la diferencia entre el máximo y mínimo
        diff = max_val - min_val

        # Asegurarse de que la diferencia no sea cero para evitar división por cero
        if diff == 0:
            diff = 1

        # Aplicar la normalización lineal a cada píxel de la imagen
        adjusted_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                pixel_val = image[i, j]
                adjusted_val = ((pixel_val - min_val) * 255 / diff).astype(np.uint8)
                adjusted_image[i, j] = adjusted_val

        return adjusted_image

    def log_contrast(self, image):
        # Obtener las dimensiones de la imagen
        height, width = image.shape

        # Definir el factor de ajuste para controlar la intensidad
        factor = 255 / math.log(256)

        # Aplicar la función de logaritmo a cada píxel de la imagen
        adjusted_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                pixel_val = image[i, j]
                adjusted_val = int(factor * math.log(1 + pixel_val))
                adjusted_image[i, j] = adjusted_val

        return adjusted_image
    
    





    # Especificacion con otra imagen

    def calc_histog_cdf(self,image):
        # Crea un array de 256 posiciones enteras
        hist = np.zeros(256, dtype=int)
        # Contar la cantidad de niveles de grises de 0-255
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i, j]] += 1
        # Calcular el histograma acumulado
        cdf = np.cumsum(hist)
        # Creacion de mapa de correspondencia
        # Normalizacion de cada valor de las y del histograma entre 0-255
        nj = (cdf - cdf.min()) * 255
        N = cdf.max() - cdf.min()
        cdf = nj / N
        # Casting de float a uint8
        cdf = cdf.astype('uint8')
        return hist,cdf
    
    def spec(self,image,ref_image):
        # Poner en un vector toda la imagen
        flat_image = image.flatten()
        # Calcular el histograma acumulado de la imagen de referencia
        _,cdf_ref_image = self.calc_histog_cdf(ref_image)
        # Crear una imagen nueva a partir del mapeo creado con la cdf
        img_new = cdf_ref_image[flat_image]
        # Redimensionar a 2D
        img_new = np.reshape(img_new,image.shape)
        return img_new


    