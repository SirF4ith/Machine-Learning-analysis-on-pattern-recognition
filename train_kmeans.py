import time
import keras
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

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
