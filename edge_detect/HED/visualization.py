import cv2
import matplotlib.pyplot as plt

out1 = cv2.imread('out1.png', 1)
out2 = cv2.imread('out2.png', 1)
out3 = cv2.imread('out3.png', 1)
out4 = cv2.imread('out4.png', 1)
out5 = cv2.imread('out5.png', 1)
out = cv2.imread('out-fused.png', 1)

plt.subplot(231)
plt.imshow(out1)
plt.subplot(232)
plt.imshow(out2,cmap='gray')
plt.subplot(233)
plt.imshow(out3,cmap='gray')
plt.subplot(234)
plt.imshow(out4,cmap='gray')
plt.subplot(235)
plt.imshow(out5,cmap='gray')
plt.subplot(236)
plt.imshow(out,cmap='gray')
plt.show()