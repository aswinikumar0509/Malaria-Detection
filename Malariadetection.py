from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# re-size the image
IMAGE_SIZE = [224,224]

train_path = "Dataset\Train"
test_path = "Dataset\Test"

# add the VGG layer to the front of VGG
vgg = VGG19(input_shape = IMAGE_SIZE + [3] , weights = "imagenet", include_top = False)

for layer in vgg.layers:
    layer.trainable = False

# usefull for getting the number of classes

folders = glob('Dataset\Train/*')

#  our layers for the model

x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use the Image Data Generator to import the image from the dataset

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('Dataset\Train',
                                                target_size = (224,224),
                                                batch_size = 32,
                                                class_mode = 'categorical'
                                                )

test_set = test_datagen.flow_from_directory('Dataset\Test',
                                            target_size = (224,224),
                                            batch_size = 32,
                                            class_mode = 'categorical'
                                            )

# fit the model
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)


import tensorflow as tf

from keras.models import load_model

model.save('model_vgg19.h5')