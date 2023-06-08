import numpy as np
import cv2
import functions
import matplotlib.pyplot as plt
import skimage

############## MAIN ################

# Add noise
image_path = 'images/skull14l.jpg'
noisyObject = functions.add_noise(image_path)
image = noisyObject.get_image()
#noisy_image = noisyObject.additive(25,25)
noisy_image = noisyObject.impulsive(1000)

# Remove noise
filtered_image = cv2.medianBlur(noisy_image, 3)

# Enhancement
unsharpObject = functions.unsharp_mask_class()
unsharp_image = unsharpObject.unsharp(r_image=image,filtered_image=filtered_image,scale_factor=1.0)

############## Visualization #############

fig, axarr = plt.subplots(2,2)
axarr[0,0].imshow(image, cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,1].imshow(noisy_image, cmap='gray')
axarr[0,1].set_title('Noisy Image')
axarr[1,0].imshow(filtered_image, cmap='gray')
axarr[1,0].set_title('Filtered Image')
axarr[1,1].imshow(unsharp_image, cmap='gray')
axarr[1,1].set_title('Unsharp Image')
plt.show()


