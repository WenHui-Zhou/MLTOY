import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils

# 数据准备导入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
"""
print(x_train.shape)
print(x_test.shape)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i+100],cmap='gray',interpolation='none')
    plt.title('Class {}'.format(y_train[i+100]))

#plt.show()
"""

#将数据有二维变成一维数据
x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)
print(x_train.shape)

# 数据归一化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = (x_train-127)/127
x_test = (x_test - 127)/127

#使用one hot encoding 来处理数据
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


model = Sequential()

#第一个隐层
model.add(Dense(units=512,input_dim=784,kernel_initializer='he_normal',
                activation='relu'))
model.add(Dropout(0.2))#dropout 防止过拟合
#第二层
model.add(Dense(units=512,input_dim=784,kernel_initializer='he_normal',
                activation='relu'))
model.add(Dropout(0.2))#dropout 防止过拟合

model.add(Dense(units=10,activation='softmax'))#输出

#训练
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=20,verbose=1,validation_split=0.05)

loss,accuracy = model.evaluate(x_test,y_test)
print('test loss: ',loss)
print('accuracy: ',accuracy)

