from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt 
import numpy as np 

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

images = lfw_people.images
b_size, H, W = images.shape

plt.imshow(images[15])

vectorized_images = images.reshape(-1, H* W)

print(vectorized_images.shape)

u, s, vt = np.linalg.svd(vectorized_images)

print(u.shape)
print(s.shape)
print(vt.shape)

# Projecting and reconstructing using only 50 principal components 
f_5 = u[:,:50] @ np.diag(s[:50]) @ vt[:50]


plt.figure()
plt.imshow(f_5[15].reshape(H, W))
plt.show()
