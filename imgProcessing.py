import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("cracked.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #grayscale
gaussBlur= cv2.GaussianBlur(gray, (5,5), 0)    #Smoothing

#Edge detection
median = np.median(gray)
low = int(0.66 * median)
high = int(1.33 * median)
edges = cv2.Canny(gaussBlur, low, high)

#Dilation and closing
kernel = np.ones((3,3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)
edges_closed = cv2.morphologyEx(
    edges_dilated, cv2.MORPH_CLOSE, kernel
)

#region filling
filled = edges_closed.copy()
h, w = filled.shape
mask = np.zeros((h+2, w+2), np.uint8)

cv2.floodFill(filled, mask, (0,0), 255)
filled_inv = cv2.bitwise_not(filled)

clean = cv2.morphologyEx(
    filled_inv, cv2.MORPH_OPEN, kernel
)
num_labels, labels = cv2.connectedComponents(clean)
output = np.zeros_like(clean)

for label in range(1, num_labels):
    mask = (labels == label).astype("uint8") * 255
    if cv2.countNonZero(mask) > 100:
        output = cv2.bitwise_or(output, mask)
overlay = img.copy()
overlay[output == 0] = [0, 0, 255]  # mark defects

print(overlay.shape)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
