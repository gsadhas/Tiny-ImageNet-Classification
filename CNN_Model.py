"""
Author : Gowdhaman Sadhasivam
E-mail : gsadha2@uic.edu

"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import cPickle

batch_size = 500
nb_classes = 100
nb_epoch = 100
data_augmentation = False

# input image dimensions
img_rows, img_cols = 64, 64
#RGB Images
img_channels = 3

#Loading training dataset
path_train = 'train100.pickle'
with open(path_train, 'rb') as handle:
        print('Opening the train pickle...')
        ptr = cPickle.load(handle)
        X_train,y_train = ptr

#Loading validation dataset
path_test = 'valData100.pickle'
with open(path_test, 'rb') as handle:
        print('Opening the test pickle...')
        pts = cPickle.load(handle)
        X_test,y_test = pts



print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
print(y_train.shape)
y_train = np.array(y_train)
y_train = np.squeeze(y_train)

print(y_train.shape)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(Y_train.shape,'y train catagory')

#Linear pile of layers
model = Sequential()

#Stacking the layers Conv 1, Conv 2, Conv 3

#Conv 1
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
print("After C1: ")
print(model.output_shape)

#Conv 2
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
print("After C2: ")
print(model.output_shape)

#Conv 3
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print("After C3: ")
print(model.output_shape)

#Fully connected layer 1
model.add(Flatten())
print("After First Flatten: ")
print(model.output_shape)
model.add(Dense(512))
print("After dense: ")
print(model.output_shape)
model.add(Activation('relu'))
model.add(Dropout(0.5))
print("Before Second Flatten: ")
print(model.output_shape)

#Fully connected layer 2
#model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Train the model using SGD + momentum
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test), shuffle=True)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

