from dotenv import load_dotenv
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from keras import layers, utils
import numpy as np
import pickle
import pyshark


load_dotenv()



with open(os.getenv("W-T-SIM_X_TE"), 'rb') as file:
    X_te = pickle.load(file)

with open(os.getenv("W-T-SIM_X_TR"), 'rb') as file:
    X_tr = pickle.load(file)

with open(os.getenv("W-T-SIM_X_VL"), 'rb') as file:
    X_vl = pickle.load(file)

with open(os.getenv("W-T-SIM_Y_TE"), 'rb') as file:
    Y_te = pickle.load(file)

with open(os.getenv("W-T-SIM_Y_TR"), 'rb') as file:
    Y_tr = pickle.load(file)

with open(os.getenv("W-T-SIM_Y_VL"), 'rb') as file:
    Y_vl = pickle.load(file)

X_train = np.array(X_tr)
y_train = np.array(Y_tr)
X_valid = np.array(X_vl)
y_valid = np.array(Y_vl)
X_test = np.array(X_te)
y_test = np.array(Y_te)


num_classes = 100
Y_train = utils.to_categorical(y_train, num_classes=num_classes)
Y_valid = utils.to_categorical(y_valid, num_classes=num_classes)
Y_test = utils.to_categorical(y_test, num_classes=num_classes)




model = keras.models.Sequential()


#This model WIP, basic one that can run the tik_tok datasets
# Input Layer
model.add(layers.Input((160, 1)))

# First Convolutional Block
model.add(layers.Conv1D(filters=8, kernel_size=8, strides=1, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ELU())
model.add(layers.MaxPooling1D(pool_size=2))

# Second Convolutional Block
model.add(layers.Conv1D(filters=16, kernel_size=8, strides=1, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ELU())
model.add(layers.MaxPooling1D(pool_size=2))

# Third Convolutional Block
model.add(layers.Conv1D(filters=32, kernel_size=4, strides=1, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ELU())
model.add(layers.MaxPooling1D(pool_size=2))

# Global Average Pooling
model.add(layers.GlobalAveragePooling1D())

# Fully Connected Layers (reduced size and dropout increased slightly)
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))


# Output Layer
model.add(layers.Dense(num_classes, activation='softmax'))

# Summary of the model
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)

#compilation
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_valid, Y_valid))

#evaluation
#test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
#print("Test accuracy: ", test_acc)
