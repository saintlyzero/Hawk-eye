import pandas as pd
import numpy as np 
import os
import keras
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint



import tensorflow as tf


tf.test.is_gpu_available(
    cuda_only = False
)


train_datagen=ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=r"data/11-5-2019/",
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


valid_datagen = ImageDataGenerator(rescale=1./255)



valid_generator = valid_datagen.flow_from_directory(
    directory=r"data/1-7-2019/validation/cleaned",
    target_size=(256, 256),
    color_mode="rgb",

    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)



img_height,img_width = 256,256 
num_classes = 7

base_model = applications.resnet50.ResNet50(weights = 'imagenet', include_top=False, input_shape= (img_height,img_width,3))


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)


from keras.optimizers import SGD, Adam
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])


#saving checkpoint
model_check=keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='auto', period=1)


from PIL import Image
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size



model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,epochs=1,callbacks=[model_check])



model=load_model('weights.01-0.54.hdf5')



model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,epochs=10,callbacks=[model_check])



model=load_model('weights.08-0.57.hdf5')



model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,epochs=10,callbacks=[model_check])


model.summary()



get_ipython().system('conda list --export > requirements.txt')


model=load_model('weights.08-0.57.hdf5')



model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,epochs=10,callbacks=[model_check])


