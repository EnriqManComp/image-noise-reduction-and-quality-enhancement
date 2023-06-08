import numpy as np
import cv2
import random
from skimage.filters import gaussian


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
    
    #def median(self,):


########################### UNSHARP MASKING ###############################

class unsharp_mask_class():

    def __init__(self):
        pass

    def unsharp(self,r_image,filtered_image,scale_factor):
        filtered_image = gaussian(filtered_image,sigma=1,mode='constant',cval=0.0)
        img = scale_factor*(r_image - filtered_image)
        unsharp_image = r_image + img
        return unsharp_image
