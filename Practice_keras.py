from keras.datasets import mnist


# 1. 데이터 적재하기
#MNIST 데이터 셋은 넘파이(numpy) 배열 형태로 케라스에 이미 적재되어있음
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)    #(60000,28,28)
print(test_images.shape)     #(10000,28,28)

# 2. 신경망 구조
from keras import models
from keras import layers
network = models.Sequential()   # initialize
network.add(layers.Dense(512, activation = 'relu', input_shape=(28*28,)))   
network.add(layers.Dense(10, activation='softmax'))

# 3. 컴파일 단계
network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# 4. 데이터셋 준비 (스케일 크기 조정, 전처리 과정)
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255     #0-1사이의 스케일로 조정

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

# 5. label 준비하기
# keras는 tensoflow에 완전히 통합되어 keras.utils import to_categorical 하면 오류 발생
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 6. 신경망 훈련
# keras에서는 fit 메서드를 호출하여 훈련데이터에 모델을 학습시킴
network.fit(train_images, train_labels, epochs = 5, batch_size = 128)












