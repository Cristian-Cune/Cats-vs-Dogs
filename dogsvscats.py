import warnings
warnings.filterwarnings('ignore',category = FutureWarning)
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D


data_dir = '/home/cristic/Documents/tf/DogsvsCats/kagglecatsanddogs_3367a/PetImages'
categories = ['Dog','Cat']
IMG_SIZE = 70
SAVE_DATA = False


def create_training_data():

	training_data = [] 

	for category in categories:
		path = os.path.join(data_dir,category)
		class_num = categories.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
				img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
				training_data.append([img_array,class_num])
			except Exception as e:
				pass
	return training_data


for category in categories:
	path = os.path.join(data_dir,category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
		img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
		#plt.imshow(img_array)
		#plt.show()
		break


X = []
y = []
if SAVE_DATA == True:
	training_data = create_training_data()

	random.shuffle(training_data)

	for feature,label in training_data:
		X.append(feature)
		y.append(label)

	X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
	y = np.array(y)

	np.save('features.npy',X)
	np.save('labels.npy',y)
else:
	X = np.load('features.npy')
	y = np.load('labels.npy')

X = X/255.0

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy' ,
	          optimizer = 'adam' ,
			  metrics = ['accuracy'])

model.fit(X,y, batch_size = 32, epochs = 3, validation_split = 0.1)