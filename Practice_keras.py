from keras.datasets import mnist


# 1. 데이터 적재하기
#MNIST 데이터 셋은 넘파이(numpy) 배열 형태로 케라스에 이미 적재되어있음
(train_image, train_labels), (test_image, test_labels) = mnist.load_data()

print(train_image.shape)    #(60000,28,28)
print(test_image.shape)     #(10000,28,28)

# 2. 신경망 구조
from keras import models
from keras import layers
network = models.Sequential()   # initialize
network.add(layers.Dense(5142, activation = 'relu', imput_shape=(28*28)))   
network.add(layers.Dense(10, activation='softmax'))













