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
#image_flattened = np.sort(noisy_image.flatten())
#plt.plot(image_flattened[:],range(len(image_flattened)))

unsharpObject = functions.unsharp_mask_class()
unsharp_image = unsharpObject.unsharp(r_image=image,filtered_image=filtered_image,scale_factor=1.0)

#plt.show()







############## Visualization #############
cv2.imshow('Image',image)
cv2.imshow('Noisy Image',noisy_image)
cv2.imshow('Filtered Image',filtered_image)
plt.imshow(unsharp_image, cmap="gray")
plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()
