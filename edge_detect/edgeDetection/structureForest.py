import cv2
import matplotlib.pyplot as plt
import numpy as np


model_path = "E:/workSpace/AI/edgeDetection/resources/model.yml.gz"
cloth_image_path = "E:/workSpace/AI/humanParsing/Sketch2Cloths/Data Production/upper garment/segCloth_version2/"
label_annotations_path = "./annotations/pixel-level/"
segImage_path = "./segCloth_version2/"


image = cv2.imread(filename,1)
if image.size == 0:
    print('cannot read file')
    exit(0)
img = np.float32(image)
img = img*(1.0/255.0)
retval = cv2.ximgproc.createStructuredEdgeDetection(model_path)
a = retval.detectEdges(img)
plt.subplot(121)
plt.imshow(image,cmap='gray')
plt.subplot(122)
plt.imshow(a,cmap= 'gray')
plt.show()
