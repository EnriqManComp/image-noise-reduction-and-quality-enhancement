import numpy as np
import cv2
from skimage.filters import gaussian
import math


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
    
    def equalization(image):
        # Using local hist
        pass

    def match_histog(self,image,ref_image):
        # Another image
        n = image.copy()
        n=n.flatten()
        histog = [0] * 256
        for intensity in n:
            histog[intensity] += 1
        return histog

    