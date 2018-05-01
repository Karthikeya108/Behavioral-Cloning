import os
import csv
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Lambda, Dropout
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = 'adam'
LOSS = 'mse'
VALIDATION_SPLIT = 0.1
IMG_ROWS, IMG_COLS = 160, 320
NB_CHANNELS = 3
NB_CLASSES = 1
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, NB_CHANNELS)

class PSAModel:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=input_shape))
        model.add(Conv2D(6, kernel_size=5, padding="valid", input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(6, kernel_size=5, padding="valid", activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(6, kernel_size=5, padding="valid", activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(6, kernel_size=5, padding="valid", activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(120))
        model.add(Dropout(0.2))
        model.add(Dense(84))
        model.add(Dense(classes))
        
        return model
        
def load_data(samples, file_path):
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        
    return samples
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = Image.open(name)
                center_angle = float(batch_sample[3])
                images.append(np.array(center_image))
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def main():
    samples = []
    load_data(samples, '../data/driving_log.csv')
    load_data(samples, '../data/driving_log1.csv')
    load_data(samples, '../data/driving_log2.csv')

    load_data(samples, '../data/driving_log4.csv')
    load_data(samples, '../data/driving_log4.csv')
    load_data(samples, '../data/driving_log4.csv')

    load_data(samples, '../data/driving_log5.csv')
    load_data(samples, '../data/driving_log5.csv')
    load_data(samples, '../data/driving_log5.csv')

    train_val_samples, test_samples = train_test_split(samples, test_size=0.1)
    train_samples, validation_samples = train_test_split(train_val_samples, test_size=0.1)
    
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
    
    model = PSAModel.build(INPUT_SHAPE, NB_CLASSES)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, 
                        validation_data=validation_generator, validation_steps=len(validation_samples)/BATCH_SIZE,
                        epochs=NB_EPOCH, verbose=VERBOSE)

    model.save('model-PSAModel-track-1a.h5')
    
    test_generator = generator(test_samples, batch_size=BATCH_SIZE)
    model.evaluate_generator(test_generator, steps=len(test_samples)/BATCH_SIZE)
    test_loss = model.evaluate_generator(test_generator, steps=len(test_samples)/BATCH_SIZE)
    print("Loss on Test Data: ",test_loss)
    
if __name__ == '__main__':
    main()