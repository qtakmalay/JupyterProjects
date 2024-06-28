import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from preprocess_function import preprocess
from data import data

#create tensorflow data pipeline
data = data.map(preprocess)
data = data.shuffle(buffer_size=5000)
data = data.cache()
data = data.batch(16)
data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

train = data.take(1982)
test = data.skip(1982).take(849)

# samples, labels = next(iter(train))
# print(f"Sample shape: {samples.shape}")  # Should match the input shape for your model
# print(f"Label shape: {labels.shape}")    # Should be (batch_size,)
# print(f"Label dtype: {labels.dtype}")    # Should be an integer type


model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(541, 257, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(21, activation='softmax'))
model.compile(optimizer='adam', loss= 'SparseCategoricalCrossentropy', metrics=['accuracy'])
# model.summary()

hist = model.fit(train, validation_data=test, epochs=5)

#graph of loss and accuracy
import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


