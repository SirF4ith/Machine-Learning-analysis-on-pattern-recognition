import time
import keras
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import math

TOTAL_CLUSTERS = 384

def get_most_common_labels(kmeans, y):
    labels = np.zeros(kmeans.n_clusters)
    for i in range(kmeans.n_clusters):
        mask = (kmeans.labels_ == i)
        if np.any(mask):
            labels[i] = np.bincount(y[mask]).argmax()
    return labels

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

kmeans = KMeans(n_clusters=TOTAL_CLUSTERS, random_state=42)

startTime_train = time.time()

kmeans.fit(x_train)

endTime_train = time.time()
print("time to train: {:.2f}".format(endTime_train-startTime_train))

common_labels = get_most_common_labels(kmeans, y_train)

startTime_test = time.time()
test_clusters = kmeans.predict(x_test)
endTime_test = time.time()
print("time to test: {:.2f}".format(endTime_test-startTime_test))

test_predictions = common_labels[test_clusters]

print(classification_report(y_test, test_predictions))

# Plotting the centroids
centroids = kmeans.cluster_centers_.reshape(TOTAL_CLUSTERS, 28, 28)

figs_to_show = 10 # Number of centroids to show
rows = math.ceil(math.sqrt(figs_to_show))
cols = math.ceil(figs_to_show / rows)

plt.figure(figsize=(28,28))
for i in range(figs_to_show):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(centroids[i], cmap='gray')
    plt.title(f'Label: {int(common_labels[i])}')
    plt.axis('off')
plt.show()