import numpy as np
import noise
import enhancement
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
from skimage.filters import gaussian

############## MAIN ################

'''

 # Cargar la imagen
image = cv2.imread('original2.jpg', 0)  # Lee la imagen en escala de grises

# Crear una instancia de la clase histog
histogram = enhancement.histog()

# Aplicar la equalización del histograma
equalized_image = histogram.histogram_equalization(image)

# Aplicar la función de raíz cuadrada
sqrt_contrast_image = histogram.sqrt_contrast(image)

# Aplicar la normalización lineal
linear_contrast_image = histogram.linear_contrast(image)

# Aplicar la función de logaritmo
log_contrast_image = histogram.log_contrast(image)

# Imprimir las imágenes resultantes o realizar cualquier otro análisis necesario
print("Imagen original:")
cv2.imshow("Original", image)

print("Imagen con histograma equalizado:")
cv2.imshow("Equalized", equalized_image)

print("Imagen con contraste mediante raíz cuadrada:")
cv2.imshow("Square Root Contrast", sqrt_contrast_image)

print("Imagen con contraste lineal:")
cv2.imshow("Linear Contrast", linear_contrast_image)

print("Imagen con contraste logarítmico:")
cv2.imshow("Log Contrast", log_contrast_image)

# Esperar a que se presione una tecla para cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


########### NOISE REMOVING ##############

# Image paths
IMAGE_PATH = 'original2.jpg'
REFERENCE_IMAGE_PATH = 'ref.jpg'

# Read images
target_image = cv2.imread(IMAGE_PATH,0)/ 255.0
ref_image = cv2.imread(REFERENCE_IMAGE_PATH,0) /255.0




# Add noise
noisyObject = noise.noise()
#noisy_image,noise = noisyObject.additive(target_image,0,10)
noisy_image = noisyObject.impulsive(target_image,1000)

# Remove noise
#result_image = noisyObject.rem_impulsive_noise(noisy_image)
result_image2 = noisyObject.rem_impulsive_noise_unsharp(noisy_image)
#target_result_diff = target_image - result_image


#result_image2 = noisyObject.rem(noisy_image)
#noisy_result_image2 = noisy_image - result_image2

#removeObject = noise.remove_noise()
#filtered_image = removeObject.rem(noisy_image)
#filtered_image = removeObject.remove(noisy_image,5,25)

# Enhancement
#unsharpObject = enhancement.unsharp_mask_class()
#unsharp_image = unsharpObject.unsharp(r_image=noisy_image,filtered_image=filtered_image,scale_factor=100.)

# Histogram
#histObject = enhancement.histog()

#imagen = histObject.spec(image,ref_image)
#hist,_ = histObject.calc_histog_cdf(imagen)
#imagen = histObject.histogram_equalization(image)
#hist2,cdf2 = histObject.calc_histog(imagen)
#imagen = histObject.map(image,cdf1)
#cv2.imwrite('spec_image.jpg',imagen)

#imagen = histObject.especificar_niveles_grises(image, ref_image)

############## Visualization #############

fig, axarr = plt.subplots(1,2)
axarr[0].imshow(target_image, cmap='gray')
axarr[0].set_title('Target Image')
#axarr[1].imshow(noisy_image, cmap='gray')
#axarr[1].set_title('Noisy image')
#axarr[1].imshow(gaussian_img, cmap='gray')
#axarr[1].set_title('Gaussian Image')
#axarr[1].imshow(res, cmap='gray')
#axarr[1].set_title('Mask')
axarr[1].imshow(result_image2, cmap='gray')
axarr[1].set_title('Unsharp Mask')
#axarr[1,0].imshow(result_image2, cmap='gray')
#axarr[1,0].set_title('Remove Gaussian Additive Image')
#axarr[1,1].imshow(noisy_result_image2, cmap='gray')
#axarr[1,1].set_title('Noisy Image - Result Image')
plt.show()
#lt.bar(range(256), hist)
#plt.title("Histograma de imagen original")
#plt.xlabel("Niveles de grises")
#plt.ylabel("Frecuencia de pixeles")
#plt.show()
#plt.bar(range(256), cdf2)
#plt.show()

