from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

# 数据准备导入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
model = load_model('mnist.h5')

plt.imshow(x_test[128],cmap = 'gray')
plt.show()

x_test = x_test.reshape(len(x_test),-1)

x_test = x_test.astype('float32')
x_test = (x_test-127)/127

y_test = np_utils.to_categorical(y_test,10)

#print(x_test[0])
print(x_test[0].reshape(1,-1).shape)
data = x_test[128].reshape(1,-1)
result = model.predict(data)
print(result)
index =np.where(result == np.max(result))
print(index[1][0])




#loss,accuracy = model.evaluate(x_test,y_test)
#print('loss: ',loss)
#print('accuracy: ',accuracy)



