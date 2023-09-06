import PIL
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import  AveragePooling2D
from keras.layers.core import Dropout
import warnings
import os
import pandas as pd
from PIL import Image
warnings.filterwarnings('ignore')

vai_a_percorso_file = os.chdir(r'C:\Users\gugli\PycharmProjects\AI_and_ML\Self_driving_car_project')
percorso_file = r'C:\Users\gugli\PycharmProjects\AI_and_ML\Self_driving_car_project'


def genera_valori(percorso_file):
    vai_a_percorso_file
    df = pd.read_csv('driving_log.csv')
    valori = []

    for i in range(len(df)-6520):
        a = []
        n = 0
        for elemento in df.iloc[i]:
            if n <= 2:
                f = str(elemento).split('\\')
                g = percorso_file + '\\' + f[-2] + '\\' + f[-1]
                n += 1
            else:
                g = elemento
            a.append(g)
        valori.append(a)
    return valori

valori = genera_valori(percorso_file)
train_samples, validation_samples = train_test_split(valori, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                print("opening: " + batch_sample[0])
                image = Image.open(batch_sample[0])
                angle = float(batch_sample[3])
                image = np.array(image)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                image = Image.fromarray(image)
                image = np.array(image.resize((64,64), PIL.Image.NEAREST))

                images.append(image[:,:, 1, None])
                angles.append(angle)

                image_flip = image[:, ::-1]
                images.append(image_flip[:,:, 1, None])
                angles.append(angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


train_generator = generator(shuffle(train_samples), batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Cropping2D(cropping=((14, 5), (0, 0)), input_shape=(64, 64, 1)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, kernel_size=5, strides=2, border_mode='same', name='conv1'))
model.add(Activation('relu'))

model.add(Convolution2D(48,kernel_size=5, strides=2, border_mode='same', name='conv2'))
model.add(Activation('relu'))

model.add(Convolution2D(72, kernel_size=5, strides=2,border_mode='same', name='conv3'))
model.add(Activation('relu'))

model.add(Convolution2D(96, 1, 3, border_mode='same', name='conv4'))
model.add(Convolution2D(96, 3, 3, border_mode='same', name='conv5'))
model.add(AveragePooling2D(pool_size=(1, 3)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

val_samples = len(validation_samples)
nb_max_epoch = 3

for j in range(3):
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         nb_val_samples=val_samples, nb_epoch=1,
                                         verbose=1)



