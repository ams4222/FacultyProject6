from dotenv import load_dotenv
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from keras import layers, utils
import numpy as np
import pickle


load_dotenv()

with open(os.getenv("UNDEFENDED_X_TE"), 'rb') as file:
    X_te = pickle.load(file)

with open(os.getenv("UNDEFENDED_X_TR"), 'rb') as file:
    X_tr = pickle.load(file)

with open(os.getenv("UNDEFENDED_X_VL"), 'rb') as file:
    X_vl = pickle.load(file)

with open(os.getenv("UNDEFENDED_Y_TE"), 'rb') as file:
    Y_te = pickle.load(file)

with open(os.getenv("UNDEFENDED_Y_TR"), 'rb') as file:
    Y_tr = pickle.load(file)

with open(os.getenv("UNDEFENDED_Y_VL"), 'rb') as file:
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


#This model WIP, basic one that can run the tik_tok datasets
model = keras.Sequential()

model.add(layers.Input((160,1)))

model.add(layers.Conv1D(filters=32, kernel_size=16, strides=1))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.ELU())
model.add(layers.MaxPooling1D())

model.add(layers.Conv1D(filters=16, kernel_size=8, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.ELU())
model.add(layers.MaxPooling1D())

model.add(layers.Conv1D(filters=16, kernel_size=8, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.ELU())
model.add(layers.MaxPooling1D())

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Dense(num_classes, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.65))

model.add(layers.Dense(num_classes, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Activation('softmax', name="softmax"))


#compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training
model.fit(X_train, Y_train, epochs=10, batch_size=128, validation_data=(X_valid, Y_valid))

#evaluation
#test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
#print("Test accuracy: ", test_acc)
