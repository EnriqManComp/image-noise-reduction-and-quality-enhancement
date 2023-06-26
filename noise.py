import numpy as np
import cv2
from skimage.filters import gaussian
import math
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import gaussian


class noise():
    
    def __init__(self):
        pass

    def gauss(self,image):
        return gaussian(image,sigma=2,mode='constant',cval=0.0)
    
    def unsharp(self,image,C):
        gauss_img = self.gauss(image)
        mask = C*(image - gauss_img)
        return image + mask

               
    def impulsive(self,image,num_pixels):
        # Ruido impulsivo
        noisy_image = np.copy(image)
        # Generar coordenadas aleatorias para los píxeles afectados por el ruido
        # ponerla con prob
        x_coords = np.random.randint(0, image.shape[0], size=num_pixels)
        y_coords = np.random.randint(0, image.shape[1], size=num_pixels)
        # Asignar valores de sal o pimienta a los píxeles seleccionados
        for i in range(len(x_coords)):
            t = np.random.normal(0, 10)
            while t < 0 or t > 1:
                t = np.random.normal(0, 10)
            if t <= 0.5:
                noisy_image[x_coords[i],y_coords[i]] = 0  # Pepper
            else:
                noisy_image[x_coords[i],y_coords[i]] = 255  # Sal
        return noisy_image

    def additive(self,image,mu,sigma):
        # Ruido aditivo
        noise = np.random.normal(mu,sigma,image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image,noise
    
    def EV(self,kernel,epsilon):
        # Get EV pixels
        EV_list = []
        pos_central_px = kernel.shape[0]//2
        central_px = kernel[pos_central_px,pos_central_px]
        limit_inf = central_px - epsilon
        limit_sup = central_px + epsilon
        for k in range(kernel.shape[0]):
            for l in range(kernel.shape[1]):
                if k != pos_central_px and l != pos_central_px:
                    if kernel[k,l] >= limit_inf and kernel[k,l] <= limit_sup:
                        EV_list.append(kernel[k,l])
        return EV_list
    
    def rem_additive_noise(self,noisy_image,sigma):
        # Definir Imagen nueva
        image_copy = np.copy(noisy_image)
        # Pasado de Kernel 5x5 sobre la imagen
        for i in range(4):
            for n in range(2,image_copy.shape[0]-2,1):
                for m in range(2,image_copy.shape[1]-2,1):
                    kernel = noisy_image[n-2:n+3,m-2:m+3].copy()
                    epsilon = 1.5*sigma
                    EV_list = self.EV(kernel,epsilon)
                    
                    if len(EV_list) > 5:
                        image_copy[n,m] = np.mean(EV_list)
        return image_copy

    def rem_impulsive_noise(self,noisy_image):
        # Definir Imagen nueva
        image_copy = np.copy(noisy_image)
        # Pasado de Kernel 5x5 sobre la imagen
        for i in range(1):
            for n in range(2,image_copy.shape[0]-2,1):
                for m in range(2,image_copy.shape[1]-2,1):
                    kernel = noisy_image[n-2:n+3,m-2:m+3].copy()
                    EV_list = self.EV(kernel,20)
                    if len(EV_list) < 3:
                        neighbors_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1],image_copy[n-1,m-1],image_copy[n-1,m+1],image_copy[n+1,m-1],image_copy[n+1,m+1]]                                       
                        neighbors_values.sort()
                        image_copy[n,m] = np.median(neighbors_values)
                        kernel = noisy_image[n-2:n+3,m-2:m+3].copy()

        return image_copy
    
    def rem_impulsive_noise_unsharp(self,noisy_image):
        # Definir Imagen nueva
        image_copy = np.copy(noisy_image)
        # Pasado de Kernel 5x5 sobre la imagen
        for i in range(1):
            for n in range(2,image_copy.shape[0]-2,1):
                for m in range(2,image_copy.shape[1]-2,1):
                    kernel = noisy_image[n-2:n+3,m-2:m+3].copy()
                    EV_list = self.EV(kernel,20)
                    EV_list.sort()
                    median = np.median(EV_list)
                    if len(EV_list) < 3:
                        neighbors_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1],image_copy[n-1,m-1],image_copy[n-1,m+1],image_copy[n+1,m-1],image_copy[n+1,m+1]]                                       
                        neighbors_values.sort()
                        image_copy[n,m] = np.median(neighbors_values) 
                    else:
                        C = 0.5
                        #mask = self.function_enhancement(image_copy[n,m] - np.median(EV_list))
                        mask = C*(image_copy[n,m] - np.median(EV_list))
                        image_copy[n,m] = image_copy[n,m] + mask
        return image_copy



    def function_enhancement(self,x):
        C = 0.1
        sigma = 25
        first_term = np.power(x,2)
        second_term = np.exp(-1*np.abs(x)/sigma)
        return C*first_term*second_term
    
    def rem(self,image):
        dim = image.shape
        image_copy = np.copy(image)
        for i in range(1):
            for n in range(2,dim[0]-2,1):
                for m in range(2,dim[1]-2,1):
                    kernel = image[n-2:n+3,m-2:m+3].copy()
                    EV_list = self.EV(kernel,15)
                    EV_list.sort()
                    if len(EV_list) < 5:                        
                        image_copy[n,m] = np.mean(EV_list)
                    else:
                        if len(EV_list) == 0:
                            #mask = self.function_enhancement(image_copy[n,m])
                            mask = 0.1*(image_copy[n,m])
                        else:
                            #mask = self.function_enhancement(image_copy[n,m] - np.mean(EV_list))
                            mask = 0.1*(image_copy[n,m] - np.mean(EV_list))
                        
                        image_copy[n,m] = image_copy[n,m] + mask
                                        
                        
                    


        return image_copy

    
    
########################### REMOVE NOISE ##################################

class remove_noise():

    def __init__(self):
        pass

    def EV(self,kernel,epsilon,pos_central_px):
        # Get EV pixels
        EV_list = []
        central_px = kernel[pos_central_px,pos_central_px]
        
        for k in range(kernel.shape[0]):
            for l in range(kernel.shape[1]):
                if k != pos_central_px and l != pos_central_px:
                    if kernel[k,l] > (central_px - epsilon) and kernel[k,l] <= (central_px + epsilon):
                        EV_list.append(kernel[k,l])
        return EV_list
    
    def rem_additive_noise(self,noisy_image):
        # Definir Imagen nueva
        image_copy = np.copy(noisy_image)
        for n in range(2,image_copy.shape[0]-2,1):
            for m in range(2,image_copy.shape[1]-2,1):
                kernel = noisy_image[n-2:n+3,m-2:m+3].copy()
                EV_list = self.EV(kernel,15,2)
                if len(EV_list) < 2:
                        neighbors_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1],image_copy[n-1,m-1],image_copy[n-1,m+1],image_copy[n+1,m-1],image_copy[n+1,m+1]]                                       
                        image_copy[n,m] = np.uint8(np.mean(neighbors_values))                    
        return image_copy




        
    
    def remove(self,image,threshold,epsilon):
        image_copy = np.copy(image)
        for i in range(1):            
            for n in range(1,image_copy.shape[0]-1,1):
                for m in range(1,image_copy.shape[1]-1,1):
                    kernel = image_copy[n-1:n+2,m-1:m+2].copy()
                    EV_list = self.EV(kernel,epsilon)
                    if len(EV_list) < threshold:
                        ##### PROBAR INCREMENTAR CANTIDAD DE VECINOS 
                        neighbors_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1],image_copy[n-1,m-1],image_copy[n-1,m+1],image_copy[n+1,m-1],image_copy[n+1,m+1]]
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
        
        
    def function_enhancement(self,x):
        C = 0.1
        sigma = 25
        first_term = np.power(x,2)
        second_term = np.exp(-1*np.abs(x)/sigma)
        return C*first_term*second_term



    def rem(self,image):
        dim = image.shape
        image_copy = np.copy(image)
        for i in range(1):
            for n in range(2,dim[0]-2,1):
                for m in range(2,dim[1]-2,1):
                    kernel = image[n-2:n+3,m-2:m+3].copy()
                    EV_list = self.EV(kernel,15)
                    #neighbors_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1],image_copy[n-1,m-1],image_copy[n-1,m+1],image_copy[n+1,m-1],image_copy[n+1,m+1]]                                       
                    neighbors_values.sort()
                    if len(EV_list) < 2:                        
                        image_copy[n,m] = np.median(neighbors_values)
                    else:
                        mask = self.function_enhancement(image_copy[n,m] - np.median(neighbors_values))
                        #mask = 0.1*(image_copy[n,m] - np.median(neighbors_values))
                        image_copy[n,m] = image_copy[n,m] + mask
                        



                    
                    
                        
                    


        return image_copy





