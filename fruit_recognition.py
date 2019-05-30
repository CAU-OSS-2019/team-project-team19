import numpy as np 
import keras 
import os
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D
from keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
from PIL import Image

fruit_list = ['Pear', 'Strawberry', 'Lemon', 'Cherry Rainier', 'Apple Braeburn', 'Mango', 'Banana', 'Tomato Cherry Red', 'Orange', 'Peach']
x_train = []
y_train = []
for i in range(10):
    fruit = fruit_list[i]
    for imageFile in os.listdir('../train/' + fruit):
        img = np.array(Image.open('../train/' + fruit + '/' + imageFile))
        x_train.append(img)
        y_train.append(i)
x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = y_train.reshape(-1,1)

num_classes = 10
epochs = 20

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
x_train = x_train.astype('float32')

x_train /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)

model_cnn = Sequential()

model_cnn.add(Conv2D(48, (3, 3), padding='same',
                                 input_shape=x_train.shape[1:]))
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))

model_cnn.add(Conv2D(80, (3, 3), padding='same',
                                 input_shape=x_train.shape[1:]))
model_cnn.add(Activation('relu'))
model_cnn.add(Conv2D(80, (3, 3), padding='same',
                                 input_shape=x_train.shape[1:]))
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))

model_cnn.add(Conv2D(128, (3, 3), padding='same',
                                 input_shape=x_train.shape[1:]))
model_cnn.add(Activation('relu'))
model_cnn.add(GlobalMaxPooling2D())
model_cnn.add(Dropout(0.25))

model_cnn.add(Dense(500))
model_cnn.add(Activation('relu'))
model_cnn.add(Dropout(0.25))
model_cnn.add(Dense(num_classes))
model_cnn.add(Activation('softmax'))
model_cnn.summary()
opt = keras.optimizers.Adam(lr=0.0001)

# train the model using RMSprop
model_cnn.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])

print("train____________")
model_cnn.fit(x_train, y_train,epochs=epochs,batch_size=128,)

# save model & weight
model_json = model_cnn.to_json()
with open("model_cnn.json", "w") as json_file : 
    json_file.write(model_json)
model_cnn.save_weights("model_cnn.h5")
print("Saved model")

# load model & weight
json_file = open("model_cnn.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_cnn.h5")
print("Loaded model")

# evaluation
loss,acc=model_cnn.evaluate(x_train,y_train)
print("loss=",loss)
print("accuracy=",acc)





