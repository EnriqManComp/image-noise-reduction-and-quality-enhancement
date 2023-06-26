import numpy as np








class noise():
    
    def __init__(self):
        pass

    def unsharp(self,image,EV_list,C):
        mask = C*(image - np.median(EV_list))
        return image + mask
    
    def function_enhancement(self,x):
        C = 0.1
        sigma = 25
        first_term = np.power(x,2)
        second_term = np.exp(-1*np.abs(x)/sigma)
        return C*first_term*second_term

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
                noisy_image[x_coords[i],y_coords[i]] = 0.  # Pepper
            else:
                noisy_image[x_coords[i],y_coords[i]] = 1.  # Salt
        return noisy_image

    def additive(self,image,mu,sigma):
        # Ruido aditivo
        noise = np.random.normal(mu,sigma,image.shape)
        noisy_image = image + noise
        #noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
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
    
    def rem_impulsive_noise(self,noisy_image):
        # Definir Imagen nueva
        image_copy = np.copy(noisy_image)
        # Pasado de Kernel 5x5 sobre la imagen
        for i in range(1):
            for n in range(2,image_copy.shape[0]-2,1):
                for m in range(2,image_copy.shape[1]-2,1):
                    kernel = noisy_image[n-2:n+3,m-2:m+3].copy()
                    EV_list = self.EV(kernel,(20/255.0))
                    if len(EV_list) < 3:
                        neighbors_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1],image_copy[n-1,m-1],image_copy[n-1,m+1],image_copy[n+1,m-1],image_copy[n+1,m+1]]                                       
                        neighbors_values.sort()
                        image_copy[n,m] = np.median(neighbors_values)                        
        return image_copy
    

    def rem_impulsive_noise_unsharp(self,noisy_image):
        # Definir Imagen nueva
        image_copy = noisy_image.copy()
        # Pasado de Kernel 5x5 sobre la imagen
        for i in range(1):
            for n in range(2,image_copy.shape[0]-2,1):
                for m in range(2,image_copy.shape[1]-2,1):
                    kernel = np.copy(image_copy[n-2:n+3,m-2:m+3])
                    EV_list = self.EV(kernel,(20/255.))
                    EV_list.sort()
                    if len(EV_list) < 3:
                        neighbors_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1],image_copy[n-1,m-1],image_copy[n-1,m+1],image_copy[n+1,m-1],image_copy[n+1,m+1]]                                       
                        neighbors_values.sort()
                        image_copy[n,m] = np.median(neighbors_values)                                                
                    else:
                        kernel = image_copy[n-2:n+3,m-2:m+3].copy()
                        unsharp_kernel = self.unsharp(kernel,EV_list,0.01)                        
                        image_copy[n-2:n+3,m-2:m+3] = unsharp_kernel.copy()                                            
        return image_copy



    
    
    
    
    
