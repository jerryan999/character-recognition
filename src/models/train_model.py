# encoding: utf-8
import numpy as np
#from keras.callbacks import callbacks, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.models import load_model
import os
import sys
import cv2
from time import time
from pathlib import Path

# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[2]


APPEARED_LETTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z'
]

CAPTCHA_TO_CATEGORY = dict(zip(APPEARED_LETTERS, range(len(APPEARED_LETTERS))))

height, width, channel = 40, 40, 1
outout_cate_num = 33
batch_size = 50
epochs = 100

random_seed = 2
X, y = [], []
for char in APPEARED_LETTERS:
	path = 'data/processed/{}'.format(char)
	for img in  os.listdir(path):
		if not img.endswith('jpg'):
			continue
		img_gray = cv2.imread(path+'/'+img,cv2.IMREAD_GRAYSCALE)
		img_ = np.expand_dims(img_gray,axis=2)
		X.append(img_)     # 增加一个dimension

		y_ = to_categorical(CAPTCHA_TO_CATEGORY[char], num_classes = len(APPEARED_LETTERS))
		y.append(y_)

# convert list to array 
X = np.stack(X, axis=0)
y = np.array(y)
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.1, random_state=random_seed)

# Set the CNN model 
# my CNN architechture is In -> [
#	[Conv2D->relu]*2 -> MaxPool2D  -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
model = Sequential()
model.add(Conv2D(filters = 32,kernel_size= (5,5),padding = 'Same',
	activation ='relu', input_shape = (height, width,channel)))
model.add(Conv2D(filters = 32,kernel_size= (5,5),padding = 'Same',
	activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
	activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
	activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(outout_cate_num, activation = "softmax"))

# look around the nn structure
model.summary()

# serialize model to JSON
#  the keras model which is trained is defined as 'model' in this example
model_json = model.to_json()
with open("models/model_num.json", "w") as json_file:
    json_file.write(model_json)


# Define the optimizer
# Set a learning rate annealer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                    patience=3, 
                                    verbose=1, 
                                    factor=0.5, 
                                    min_lr=0.01)


# With data augmentation to prevent overfitting (accuracy 0.99286)
datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
		zoom_range = 0.1, # Randomly zoom image 
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=False,  # randomly flip images
		vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, 
					  batch_size=batch_size),
                      epochs = epochs, 
                      validation_data = (X_val,Y_val),
                      verbose = 2, 
                      steps_per_epoch=X_train.shape[0] // batch_size, 
                      callbacks=[
	                      learning_rate_reduction])

# serialize weights to HDF5
model.save_weights("models/model_num-{}.h5".format(time()))

