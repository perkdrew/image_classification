
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras import regularizers
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# Here we just make sure the image format is as desired. This will make the feature (x)
# data - i.e. the RGB pixel values - for each image have the shape 3x32x32.
if K.backend()=='tensorflow':
    K.tensorflow_backend.set_image_dim_ordering('th')



def getModel(CIF):
    """Specify the CNN architecture"""
    weight_decay = 1e-4
    model = Sequential()
    #FEATURE DETECTION
    #First layer
    model.add(Conv2D(32, kernel_size=(3,3), 
                     input_shape=(3,32,32),
                     kernel_regularizer=regularizers.l2(weight_decay),
                     padding = 'same', 
                     activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    #Second layer
    model.add(Conv2D(32, kernel_size=(3,3), 
                     padding = 'same', 
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    #Third layer
    model.add(Conv2D(64, kernel_size=(3,3), 
                     padding = 'same', 
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    #Fourth layer
    model.add(Conv2D(64,kernel_size=(3,3), 
                     padding = 'valid', 
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Flatten())
    #CLASSIFICATION
    #Fully connected layer
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    #Output layer w/ softmax
    model.add(Dense(CIF.num_classes, activation='softmax'))

    #Compile the mode
    opt_rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['acc'])
    #model.summary()
    return model
  
  
def fitModel(model, CIF):
    """Fit the model to data"""
    early = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5, 
                              mode='auto')
    
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', 
                                   save_best_only=True)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.1,
                                  patience=2, 
                                  min_lr=1e-8)
    
    model.fit(CIF.x_train, CIF.y_train,
          batch_size=128,
          epochs=100,
          validation_data=(CIF.x_valid, CIF.y_valid), 
          callbacks=[checkpointer, reduce_lr, early])
    model.load_weights('/tmp/weights.hdf5')

    return model


def runImageClassification(getModel=None,fitModel=None,seed=7):
    # Fetch data. You may need to be connected to the internet the first time this is done.
    # After the first time, it should be available in your system. On the off chance this
    # is not the case on your system and you find yourself repeatedly downloading the data, 
    # you should change this code so you can load the data once and pass it to this function. 
    print("Preparing data...")
    data=CIFAR(seed)
        
    # Create model 
    print("Creating model...")
    model=getModel(data)
    
    # Fit model
    print("Fitting model...")
    model=fitModel(model,data)

    # Evaluate on test data
    print("Evaluating model...")
    score = model.evaluate(data.x_test, data.y_test, verbose=0)
    print('Test accuracy:', score[1])


class CIFAR:
    def __init__(self,seed=0):
        # Get and split data
        data = self.__getData(seed)
        self.x_train_raw=data[0][0]
        self.y_train_raw=data[0][1]
        self.x_valid_raw=data[1][0]
        self.y_valid_raw=data[1][1]
        self.x_test_raw=data[2][0]
        self.y_test_raw=data[2][1]
        # Record input/output dimensions
        self.num_classes=10
        self.input_dim=self.x_train_raw.shape[1:]
         # Convert data
        self.y_train = np_utils.to_categorical(self.y_train_raw, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid_raw, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test_raw, self.num_classes)
        self.x_train = self.x_train_raw.astype('float32')
        self.x_valid = self.x_valid_raw.astype('float32')
        self.x_test = self.x_test_raw.astype('float32')
        self.x_train  /= 255
        self.x_valid  /= 255
        self.x_test /= 255
        # Class names
        self.class_names=['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    def __getData (self,seed=0):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return self.__shuffleData(x_train,y_train,x_test,y_test,seed)
    
    def __shuffleData (self,x_train,y_train,x_test,y_test,seed=0):
        tr_perc=.75
        va_perc=.15
        x=np.concatenate((x_train,x_test))
        y=np.concatenate((y_train,y_test))
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)
        indices = np.random.permutation(len(x))
        tr=round(len(x)*tr_perc)
        va=round(len(x)*va_perc)
        self.tr_indices=indices[0:tr]
        self.va_indices=indices[tr:(tr+va)]
        self.te_indices=indices[(tr+va):len(x)]
        x_tr=x[self.tr_indices,]
        x_va=x[self.va_indices,]
        x_te=x[self.te_indices,]
        y_tr=y[self.tr_indices,]
        y_va=y[self.va_indices,]
        y_te=y[self.te_indices,]
        return ((x_tr,y_tr),(x_va,y_va),(x_te,y_te))

    # Print figure with 10 random images, one from each class
    def showImages(self):
        fig = plt.figure(figsize=(8,3))
        for i in range(self.num_classes):
            ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
            idx = np.where(self.y_valid_raw[:]==i)[0]
            features_idx = self.x_valid_raw[idx,::]
            img_num = np.random.randint(features_idx.shape[0])
            im = np.transpose(features_idx[img_num,::],(1,2,0))
            ax.set_title(self.class_names[i])
            plt.imshow(im)
        plt.show()

