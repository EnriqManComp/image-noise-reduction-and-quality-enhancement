import numpy as np
import functions
import matplotlib.pyplot as plt

############## MAIN ################

# Add noise
image_path = 'images/CCD.jpg'
noisyObject = functions.add_noise(image_path)
image = noisyObject.get_image()
#noisy_image = noisyObject.additive(25,25)
noisy_image = noisyObject.impulsive(1000)

removeObject = functions.remove()
filtered_image = removeObject.median(noisy_image)

# Enhancement
unsharpObject = functions.unsharp_mask_class()
unsharp_image = unsharpObject.unsharp(r_image=image,filtered_image=filtered_image,scale_factor=200.)

# Histogram
histObject = functions.histog()
hist = histObject.match_histog(image,image)





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
plt.bar(range(256), hist)
plt.show()

