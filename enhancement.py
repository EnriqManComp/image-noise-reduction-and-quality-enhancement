import numpy as np






######################## HISTOGRAM TECHNIQUES #############################

class histog():

    def __init__(self):
        pass
    
    def histogram_equalization(self, image):
        hist,_ = self.calc_histog_cdf(image)
        
        cdf = np.cumsum(hist)
        
        # Normalizar el CDF al rango [0, 1]
        cdf_normalized = cdf / float(np.sum(hist))

        # Aplicar la función exponencial para ajustar el histograma
        gamma = 1.5  # Parámetro de ajuste
        adjusted_cdf = cdf_normalized ** gamma

        # Mapear los valores de píxeles originales a los nuevos valores ajustados
        adjusted_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel_value = image[i, j]
                adjusted_pixel_value = int(adjusted_cdf[pixel_value] * 255)
                adjusted_image[i, j] = adjusted_pixel_value

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
    
    def sqrt_contrast(self, image):
        # Obtener las dimensiones de la imagen
        height, width = image.shape

        # Aplicar la función de raíz cuadrada a cada píxel de la imagen
        adjusted_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                pixel_val = image[i, j]

                # Ajustar los parámetros para obtener un mejor resultado
                adjusted_val = int(np.sqrt(pixel_val) * 0.5)  # Multiplicar por un factor mayor para aumentar el contraste

                # Limitar el valor ajustado para que esté en el rango de 0 a 255
                adjusted_val = max(0, min(255, adjusted_val))

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


    