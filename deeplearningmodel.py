import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

#loading the dataset
data = pandas.read_csv('ourdataset.csv')

#features and lables
x = data.drop(columns=['label'])
y = data['label']

#using encoding for labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

#normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(y_categorical.shape[1], activation='softmax')
])

#compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

#evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy: ", test_acc)

#mapping
def predict_traffic_type(sample):
    prediction = model.predict(numpy.array([sample]))
    class_index = numpy.argmax(prediction)
    return label_encoder.inverse_transform([class_index])[0]
