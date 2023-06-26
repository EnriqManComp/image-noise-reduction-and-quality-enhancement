import numpy as np
import cv2
from skimage.filters import gaussian
import math
from PIL import Image


############## NOISE EVENTS ################
 
class add_noise():
    
    def __init__(self,image_path):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def get_image(self):
        return self.image        
        
    def impulsive(self,num_pixels):
        # Ruido impulsivo
        noisy_image = np.copy(self.image)
        # Generar coordenadas aleatorias para los píxeles afectados por el ruido
        x_coords = np.random.randint(0, self.image.shape[0], size=num_pixels)
        y_coords = np.random.randint(0, self.image.shape[1], size=num_pixels)
        # Asignar valores de sal o pimienta a los píxeles seleccionados
        for i in range(len(x_coords)):
            t = np.random.normal(0.5, 0.1)
            while t < 0 or t > 1:
                t = np.random.normal(0.5, 0.1)
            if t <= 0.5:
                noisy_image[x_coords[i],y_coords[i]] = 0  # Pepper
            else:
                noisy_image[x_coords[i],y_coords[i]] = 255  # Sal
        return noisy_image

    def additive(self,mu,sigma):
        # Ruido aditivo
        noise = np.random.normal(mu,sigma,self.image.shape)
        noisy_image = self.image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
########################### REMOVE NOISE ##################################

class remove():

    def __init__(self):
        pass

    def EV(self,kernel,epsilon):
        # Get EV pixels
        EV_list = []
        pos_central_px = math.ceil(kernel.shape[0]/2)
        central_px = kernel[pos_central_px,pos_central_px]
        for k in range(kernel.shape[0]):
            for l in range(kernel.shape[1]):
                if k != pos_central_px and l != pos_central_px:
                    if kernel[k,l] >= central_px - epsilon:
                        if kernel[k,l] <= central_px + epsilon:
                            EV_list.append(kernel[k,l])
        return EV_list
    
    def median(self,image,threshold,epsilon):
        image_copy = np.copy(image)
        for i in range(1):            
            for n in range(1,image_copy.shape[0]-1,1):
                for m in range(1,image_copy.shape[1]-1,1):
                    kernel = image_copy[n-1:n+2,m-1:m+2].copy()
                    EV_list = self.EV(kernel,epsilon)
                    if len(EV_list) < threshold:
                        ##### PROBAR INCREMENTAR CANTIDAD DE VECINOS 
                        neighbors_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1]]
                        neighbors_values.sort()
                        if len(EV_list)%2 != 0 or len(EV_list) == 0:                            
                            term1 = neighbors_values[(len(neighbors_values)//2)-1] / 2
                            term2 = neighbors_values[len(neighbors_values)//2] / 2
                            median = term1 + term2
                        else:
                            median = neighbors_values[len(neighbors_values)//2]
                        image_copy[n,m] = np.int8(median)
                    #else:
                        #image_copy[n,m] = np.mean(EV_list)
            threshold +=2
                    

                    
                
                    
        return image_copy
        
        


########################### UNSHARP MASKING ###############################

class unsharp_mask_class():

    def __init__(self):
        pass

    def unsharp(self,r_image,filtered_image,scale_factor):
        filtered_image = gaussian(r_image,sigma=1,mode='constant',cval=0.0)
        img = scale_factor*(r_image - filtered_image)
        unsharp_image = r_image + img
        return unsharp_image

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
    

    def sqrt_contrast(image):
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


    def linear_contrast(image):
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
      
     

    def log_contrast(image):
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






    def calc_histog(self,imagen):
        histograma = [0] * 256
        width, height = imagen.size

        for y in range(height):
            for x in range(width):
                pixel = imagen.getpixel((x, y))
                nivel_gris = pixel  
                histograma[nivel_gris] += 1

        return histograma

    def calc_cdf(self,histograma):
        cdf = [0] * 256
        cdf[0] = histograma[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + histograma[i]

        return cdf
    def especificar_niveles_grises(self,imagen_entrada, imagen_referencia):
        
        histograma_entrada = self.calc_histog(imagen_entrada)
        cdf_entrada = self.calc_cdf(histograma_entrada)

        histograma_referencia = self.calc_histog(imagen_referencia)
        cdf_referencia = self.calc_cdf(histograma_referencia)
        
        nueva_imagen = Image.new('L', imagen_entrada.size)  # Crear nueva imagen en escala de grises
        
        width, height = imagen_entrada.size
        for y in range(height):
            for x in range(width):
                pixel_entrada = imagen_entrada.getpixel((x, y))
                nivel_gris_entrada = pixel_entrada
                nivel_gris_referencia = cdf_referencia.index(min(cdf_referencia, key=lambda x: abs(x - cdf_entrada[nivel_gris_entrada])))
                nueva_imagen.putpixel((x, y), nivel_gris_referencia)

        return nueva_imagen

