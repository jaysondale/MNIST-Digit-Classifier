'''
HANDWRITTEN DIGIT CLASSIFIER
Jayson Dale
Last edit: November 20, 2019
'''

import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

# Print a sample image from the dataset
sample_index = 1005
plt.imshow(trainX[sample_index])
print(trainY[sample_index])

print(f"Image data shape: {trainX.shape}")
print(f"Digit value list shape: {trainY.shape}")

# Keras API requires input data to be in four dimensions
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
testX = testX.reshape(testX.shape[0], 28, 28, 1)

print(f"New training data shape: {trainX.shape}")

# Now we want to normalize the RGB codes (scale from 0-255 to 0-1)
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

# Defining the model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

# Network parameters
KERNEL_SIZE = 3
POOL_SIZE = 3
HL1_SIZE = 128
EPOCHS = 10

model = Sequential()
# Add convolutional layer
model.add(Conv2D(28, input_shape=(28, 28, 1), kernel_size=(KERNEL_SIZE, KERNEL_SIZE)))
# Add pooling layer
model.add(MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)))
# Flatten pooling layer outputs
model.add(Flatten())
# Add fully connected hidden layer
model.add(Dense(HL1_SIZE, activation=tf.nn.relu))
# Add dropout
model.add(Dropout(0.2))
# Add output layer (softmax is used to compute probabilities for each output)
model.add(Dense(10, activation=tf.nn.softmax))

# Compile the model
model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')

# Train the model
print("Training model...")
model.fit(x=trainX, y=trainY, epochs=EPOCHS)

# Evaluate the model
print("Evaluating model...")
model.evaluate(testX, testY)