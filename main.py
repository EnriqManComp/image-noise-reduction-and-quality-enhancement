import numpy as np
import noise
import enhancement
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
from skimage.filters import gaussian

############## MAIN ################

# Image paths
IMAGE_PATH = 'original2.jpg'
REFERENCE_IMAGE_PATH = 'ref.jpg'

# Read images
target_image = cv2.imread(IMAGE_PATH,0)/ 255.0
ref_image = cv2.imread(REFERENCE_IMAGE_PATH,0)

# Add impulsive noise
noisyObject = noise.noise()
noisy_image = noisyObject.impulsive(target_image,800)

# Remove noise
result_image = noisyObject.rem_impulsive_noise(noisy_image)
result_image2 = noisyObject.rem_impulsive_noise_unsharp(noisy_image)

# float values to 0-255 values uint8 
image = result_image2 * 255
image = np.round(image) 
image = np.clip(image, 0, 255) 
image = image.astype(np.uint8) 

histObject = enhancement.histog()
img = histObject.sqrt_contrast(image)
#img = histObject.histogram_equalization(image)
#img = histObject.linear_contrast(image)


# Mejoramiento de contraste
#histObject = enhancement.histog()
#hist,_ = histObject.calc_histog_cdf(image)
#ref_hist,_ = histObject.calc_histog_cdf(ref_image)
#spec_image = histObject.spec(image,ref_image)
#spec_hist,_ = histObject.calc_histog_cdf(spec_image)


############## Visualization #############
plt.imshow(img, cmap='gray')
plt.show()
'''
fig, axarr = plt.subplots(2,2)
axarr[0,0].imshow(target_image, cmap='gray')
axarr[0,0].set_title('Imagen Objetivo')
axarr[0,1].imshow(noisy_image, cmap='gray')
axarr[0,1].set_title('Imagen Ruidosa')
axarr[1,0].imshow(result_image, cmap='gray')
axarr[1,0].set_title('Eliminar ruido Método 1')
axarr[1,1].imshow(result_image2, cmap='gray')
axarr[1,1].set_title('Eliminar Ruido + Realce de detalles')
plt.show()

fig, axarr = plt.subplots(2,3)
axarr[0,0].imshow(result_image2, cmap='gray')
axarr[0,0].set_title('Imagen Objetivo')
axarr[0,1].imshow(ref_image, cmap='gray')
axarr[0,1].set_title('Imagen de referencia')
axarr[0,2].imshow(spec_image, cmap='gray')
axarr[0,2].set_title('Imagen Especificada')
axarr[1,0].bar(range(256), hist)
axarr[1,0].set_title('Histograma de Imagen Objetivo')
axarr[1,1].bar(range(256), ref_hist)
axarr[1,1].set_title('Histograma de Imagen de Referencia')
axarr[1,2].bar(range(256), spec_hist)
axarr[1,2].set_title('Histograma de Imagen Especificada')
plt.show()

'''

