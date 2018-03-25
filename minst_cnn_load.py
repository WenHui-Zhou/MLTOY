from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

# 数据准备导入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
model = load_model('mnist_cnn.h5')

plt.imshow(x_test[128],cmap = 'gray')
plt.show()

x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_test = x_test.astype('float32')
x_test = (x_test-127)/127

y_test = np_utils.to_categorical(y_test,10)
loss,accuracy = model.evaluate(x_test,y_test)
print('loss: ',loss)
print('accuracy: ',accuracy)



