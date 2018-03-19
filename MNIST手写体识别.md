MNIST手写体识别
==========

前言
--

MNIST手写体数据库算得上是深度学习领域的果蝇，用它做实验十分的方便。这个task也算是我第一次，比较接近当前主流主流技术，主流方法做的一个实验。作为我的“hello world”，我将详细介绍一些相关的技术方法，以便往后回顾。

问题描述
----

MNIST数据库的组成：

 - x_train（训练集）：60000张28*28像素的图片，元素数值范围在0-255，颜色由白到黑
 - y_train（训练集的label）：60000张，每一个维度为10（代表0到9中可能）
 - x_test（测试集）：10000张28*28图片
 - y_test（测试集label）：10000张，每张10维

需要做的工作是在train set 上根据手写体以及label，train出一个可以对图片进行predict的算法。

使用工具
----

这次实验使用了keras深度学习toolkit，keras对theano及TensorFlow进行了封装，其后端符号计算部分的任务交由TensorFlow完成。

**keras的安装：** 安装后anaconda后，打开anaconda的prompt执行窗口:

1.	安装 tensorflow：`conda install tensorflow`
2.	安装keras： `conda install keras`

完成安装后前期准备工作已经做好,下面正式上手。


----------

数据处理
----

keras在其库中封装了MNIST数据库，我们可以直接引用：

    from keras.datasets import mnist
    (x_train,y_train),(x_test,y_test) = minst.load_data()

我们通过load_data()函数将数据存入了numpy narray中，这一步不用自己做了，如果拿到的是原始的mnist数据文件的话，应该是需要自己写一些解析代码的。

接下来我们对数据进行一下显示：

    for i in range(9):
	    plt.subplot(3,3,i+1) #共有3*3个子图，现在画i+1那个
	    plt.imshow(x_train[i],cmap='gray',interpolation='none')
	    plt.title('class {}'.format(y_train[i]))
	plt.show()

其中imshow（）函数将按像素描绘的方式将图像显示出来。

**将数据转化为一维：**

对于x_train，x_test来说，他们的数据维度是（60000,28,28），我们要做的是将数据变成一维即（60000,784）

    x_train = x_train.reshape(len(x_train),-1)
    x_test = x_test.reshape(len(x_test),-1)
    
**对数据进行归一化处理：**

此时我们得到的数据范围在0-255之间，我们将他们归一化到0附近，[-1,1] 之间。

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = (x_train-127)/127
    x_test = (x_test-127)/127

**one-hot encoding 处理：**

one-hot encoding（独热码），直观来说就是有多少个状态就有多少个bit，而一次只有一位为1，用于label分类时使用。引入独热码是为了防止因为label的差异引起的误差，例如1与199对模型结果存在很大影响。

编码过程即有多少中情况就有多少个位，相应的位为1表示该位被选中。

    y_train = np_utils.to_categorical(y_train,10)
    y_text = np_utils.to_categorical(y_test,10)

以上我们将数据转化为一维后，将x_train进行归一化，对y_train进行了独热码的编码，数据准备部分完成。

模型的建立
-----

keras提供了两种模型，一种为最简单的Sequential 序贯模型，一种为“functional model API”。我们使用最简单的Sequential 来创建模型。

    model = Sequential()  #创建一个模型
    #第一个隐层model.add(Dense(units=512,input_dim=784,kernel_initializer='he_normal',activation = relu))
    model.add(Dropout(0.2)) #防止过拟合

keras使用model.add()的方式添加隐层，Dense()函数确定了神经元间的连接为全连接，其参数如下：

 - units = 512  说明有512个神经元，也即输出512
 - input_dim = 784  说明有784个输入
 - kernel_initializer = 'he_normal' 使用这种初始化方式初始化，He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入
 - activation = 'relu' 使用relu函数作为激励函数
 - Dropout(0.2) 防止过拟合

**关于Dropout：**

Dropout有hinton在2012年提出，用于防止过拟合，即在每次训练的时候，让一半的特征检测器（神经元）停止工作，提高网络的泛化能力。每次训练时每个神经元有50%的概率被移除。

添加第二层：

    model.add(Dense(units = 512,activation = 'relu',kernel_initializer = 'he_normal'))
    model.add(Dropout(0.2))

输出层：

    model.add(Dense(units = 10mactivation = softmax))

训练：

    model.compile(optimizer = 'admim',loss = 'categorical_crossentroy',metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size = 64,epochs = 20,verbose = 1,validation_split = 0.05)
    loss,accuracy = model.evaluate(x_test,y_test)
    print(loss,accuracy)

以上建立以及训练完了一个程序。

保存模型：

model.save('mnist.h5')

完整代码：
-----

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
    plt.show()"""
    
    
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
    model.add(Dense(units=512,kernel_initializer='he_normal',
                    activation='relu'))
    model.add(Dropout(0.2))#dropout 防止过拟合
    
    model.add(Dense(units=10,activation='softmax'))#输出
    
    #训练
    model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=64,epochs=20,verbose=1,validation_split=0.05)
    
    loss,accuracy = model.evaluate(x_test,y_test)
    print('test loss: ',loss)
    print('accuracy: ',accuracy)
    
    model.save('mnist.h5')

训练结果
----
loss = 0.0828703 , accuracy = 0.9785

结果还是不错的。


保存的模型重新调用
---------

    from keras.models import load_model
    from keras.datasets import mnist
    from keras.utils import np_utils
    
    # 数据准备导入数据
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    model = load_model('mnist.h5')
    
    x_test = x_test.reshape(len(x_test),-1)
    
    x_test = x_test.astype('float32')
    x_test = (x_test-127)/x_test
    
    y_test = np_utils.to_categorical(y_test,10)
    
    loss,accuracy = model.evaluate(x_test,y_test)
    print('loss: ',loss)
    print('accuracy: ',accuracy)


最后
--

手写体的代码如下：

 -  [minst.py](https://github.com/WenHui-Zhou/MLTOY/blob/master/mnist.py) ： 读取数据搭建模型，以及训练模型，保存模型
 - [model_load_model.py](https://github.com/WenHui-Zhou/MLTOY/blob/master/mnist_load_model.py) ：读取模型（mnist.h5），使用模型进行预测