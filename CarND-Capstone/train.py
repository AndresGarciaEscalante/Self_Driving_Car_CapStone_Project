import pandas as pd
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.applications import InceptionV3
from keras import Sequential
from keras.layers import Activation, Cropping2D, Lambda, Conv2D, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

## importing all data into a pandas dataframe
green_paths = glob.glob("./simulator_dataset/green/*.jpg")
red_paths = glob.glob("./simulator_dataset/red/*.jpg")
yellow_paths = glob.glob("./simulator_dataset/yellow/*.jpg")
none_paths = glob.glob("./simulator_dataset/none/*.jpg")
green_df = pd.DataFrame({'image_path':green_paths, 'labels':np.array(['green']*len(green_paths))})
red_df = pd.DataFrame({'image_path':red_paths, 'labels':np.array(['red']*len(red_paths))})
yellow_df = pd.DataFrame({'image_path':yellow_paths, 'labels':np.array(['yellow']*len(yellow_paths))})
none_df = pd.DataFrame({'image_path':none_paths, 'labels':np.array(['none']*len(none_paths))})
df = pd.concat([green_df, red_df, yellow_df, none_df], ignore_index=True)
df['image_path'] = df['image_path'].str.replace('\\','/')


# one hot encode
df = pd.concat([df,pd.get_dummies(df['labels'])], axis = 1)

def get_images(image_paths):
    """
        This functions gets a numpy array of image paths and returns
        a numpy array of RGB images
    """
    images = []
    for path in image_paths:
        images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    return np.array(images)

def batch_generator(dataframe, batch_size):
    """
        This generator creates batches from all dataframe based on the batch size
        
        @params dataframe: pandas dataframe that contains all data and steering labels 
        @params batch_size: integer that sets the number of batches to which our data will be split on
        @yield batch: each iteration our generator yields a random batch
    """
    
    if type(dataframe) != pd.core.frame.DataFrame:
        raise ValueError('the input to batch_generator is not a pandas dataframe')
    while 1: # Loop forever so the generator never terminates
        temp_df = dataframe.sample(batch_size)
        # getting steering angles for the 3 cameras
        labels =  temp_df[['green','none','red', 'yellow']].values
        # getting image paths for the 3 cameras
        img_paths = np.array(temp_df['image_path'])
        # getting images from image paths
        images = get_images(img_paths)
        
        batch_Y = labels
        batch_X = images
        yield shuffle(batch_X, batch_Y)

# split data        
train_df, validation_df  = train_test_split(df,
                                            stratify=df['labels'], 
                                            test_size=0.2)

## Creating batches
batch_size = 10
val_generator = batch_generator(validation_df, batch_size)
train_generator = batch_generator(train_df, batch_size)

## inception model
inception_model = InceptionV3(include_top=False, weights='imagenet')
inception_model.trainable = False

## mymodel
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(600,800,3)))
# Start architecture here
model.add(inception_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(4))
model.add(Activation("softmax"))
model.summary()

## Run model
model_path="model.h5"

checkpoint = ModelCheckpoint(model_path, 
                              monitor= 'val_loss', 
                              verbose=1, 
                              save_best_only=True, 
                              mode= 'min', 
                              save_weights_only = False,
                              period=2)

early_stop = EarlyStopping(monitor='val_loss', 
                       mode= 'min', 
                       patience=2)


callbacks_list = [checkpoint, early_stop]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_df)/batch_size), 
            validation_data=val_generator, 
            validation_steps=np.ceil(len(validation_df)/batch_size), 
            epochs=5, verbose=1, callbacks=callbacks_list)