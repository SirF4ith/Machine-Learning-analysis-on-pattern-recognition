import time
import keras
import numpy as np
from keras import layers
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
# hyperparameters
batch_size = 128
epochs = 20

model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

startTime_train = time.time()

# model training
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_split=0.1)

endTime_train = time.time()
print("Time for training: {:.2f} seconds".format(endTime_train-startTime_train))

# model test
startTime_test = time.time()
y_pred = model.predict(x_test)
endTime_test = time.time()
print("time to test: {:.2f} seconds".format(endTime_test-startTime_test))

# transform predicitons into one-hot-coding for evaluation
for i in range(len(y_pred)):
    max_index = np.argmax(y_pred[i])
    y_pred[i] = np.zeros_like(y_pred[i])
    y_pred[i][max_index] = 1

# evaluation
print(classification_report(y_test, y_pred))

# Plotting multilabel confusion matrices
confusion_matrices = multilabel_confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

fig, axes = plt.subplots(2, 5, figsize=(25, 15))
axes = axes.ravel()

for i, cm in enumerate(confusion_matrices):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[i], values_format='d')
    axes[i].set_title(f'Ziffer {i}', fontsize=25)
    disp.im_.colorbar.remove()
    axes[i].set_xlabel('Predicted label', fontsize=13)
    axes[i].set_ylabel('True label', fontsize=13)

    for text in disp.text_.ravel():
        text.set_fontsize(20)

plt.subplots_adjust(wspace=0.20, hspace=0.1)
fig.colorbar(disp.im_, ax=axes, fraction=0.02, pad=0.04)
plt.suptitle('Multilabel Confusion Matrizen für CNN', fontsize=50)

plt.savefig('confusion_matrix_cnn.png', bbox_inches='tight')
plt.show()
