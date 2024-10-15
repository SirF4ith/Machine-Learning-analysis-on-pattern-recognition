import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
import keras
from sklearn.metrics import classification_report

TOTAL_CLUSTERS = 512

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

kmeans.fit(x_train)

common_labels = get_most_common_labels(kmeans, y_train)

test_clusters = kmeans.predict(x_test)
test_predictions = common_labels[test_clusters]

print(classification_report(y_test, test_predictions))

# saving the model
