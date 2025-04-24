from dotenv import load_dotenv
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from keras import layers, utils, preprocessing
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import Counter


load_dotenv()


'''
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


print(X_te)

X_train = np.array(X_tr)
y_train = np.array(Y_tr)
X_valid = np.array(X_vl)
y_valid = np.array(Y_vl)
X_test = np.array(X_te)
y_test = np.array(Y_te)

print(y_valid[0:50])

num_classes = 100
Y_train = utils.to_categorical(y_train, num_classes=num_classes)
Y_valid = utils.to_categorical(y_valid, num_classes=num_classes)
Y_test = utils.to_categorical(y_test, num_classes=num_classes)

print(X_valid[0:50])
'''

with open('./all_sequences.pkl', 'rb') as f:
        x_data = pickle.load(f)

with open('./all_sequences_pickle.pkl', 'rb') as f:
        y_data = pickle.load(f)


normal_data = [item[1] for item in x_data]
tor_data = [item[1] for item in y_data]



x_data = []
y_data = []

for dataset in normal_data:
    x_data.extend(dataset)

for dataset in tor_data:
    y_data.extend(dataset)


labels_chunk_1 = [0] * len(x_data)
labels_chunk_2 = [1] * len(y_data)


combined_data = x_data + y_data
combined_labels = labels_chunk_1 + labels_chunk_2


clean_data = [(float(ts), val) for ts, val in combined_data]
np_data = np.array(clean_data)
np_labels = np.array(combined_labels)

shuffled_data, shuffled_labels = shuffle(np_data, np_labels, random_state=42)

num_classes = 2

Y_data = utils.to_categorical(shuffled_labels, num_classes)


X_train, X_temp, Y_train, Y_temp = train_test_split(
    shuffled_data, shuffled_labels,
    test_size=0.3,
    random_state=42,
)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp,
    test_size=0.5,  # half of 30% = 15%
    random_state=42,
)


X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_val = np.array(Y_val)
Y_test = np.array(Y_test)

Y_train = utils.to_categorical(Y_train, num_classes)
Y_val = utils.to_categorical(Y_val, num_classes)
Y_test = utils.to_categorical(Y_test, num_classes)

print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
#print(f"X_val: {X_val.shape}, Y_val: {Y_val.shape}")
print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")

model = keras.models.Sequential()


#This model WIP, basic one that can run the tik_tok datasets
# Input Layer
model.add(layers.Input((5000, 2)))

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
#model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)

#compilation
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training


sequence_length = 5000

# Calculate the number of full sequences that can be made from the available data
num_sequences = len(X_train) // sequence_length
# Reshape X_train into sequences of 160 time steps, each with 2 features
X_train_reshaped = X_train[:num_sequences * sequence_length].reshape(-1, sequence_length, 2)
X_val_reshaped = X_train[:num_sequences * sequence_length].reshape(-1, sequence_length, 2)
# If Y_train is a single label per sample (e.g., for classification), we also reshape it.
# Assuming your Y_train is structured similarly to X_train but with 1 label per sample
Y_train_reshaped = Y_train[:num_sequences]
Y_val_reshaped = Y_train[:num_sequences]

print(X_train.shape)
print(Y_train.shape)
model.fit(X_train_reshaped, Y_train_reshaped, epochs=20, batch_size=5, validation_data=(X_val_reshaped, Y_val_reshaped))

#evaluation
#test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
#print("Test accuracy: ", test_acc)