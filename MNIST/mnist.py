from tkinter import N
import numpy as np
import time
from matplotlib import pyplot as plt
from keras.datasets import mnist #!NB: pip install tensorflow, pip install keras -> Skriv i readme.
from sklearn.neighbors import KNeighborsClassifier # NB: pip install sklearn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans

# TODO: 
# Add description of file structure and what is does
# se på performance for ulike k-verdier, se: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
# lag loop for å velge oppgave input/output greie 

# BEGIN Utilities

def flatten_split_data(x_train, y_train, x_test, y_test, n_chunks):
    """Reshapes datasets to 2D-arrays and splits them into n chunks.
    Params:
        x_train, y_train: np.ndarray
            Training dataset
        x_test, y_test: np.ndarray
            Testing dataset
        n_chunks: int
          Number of chunks to split data into
    Returns:
        Reshaped arrays
    """
    # Flatten 2D data vectors/images into 1D arrays
    nsamples_train, nx_train, ny_train = x_train.shape
    x_train = x_train.reshape((nsamples_train,nx_train*ny_train))
    nsamples_test, nx_test, ny_test = x_test.shape
    x_test = x_test.reshape((nsamples_test,nx_test*ny_test))
    # Split data
    x_train = np.split(x_train, n_chunks)
    y_train = np.split(y_train, n_chunks)
    x_test = np.split(x_test, n_chunks)
    y_test = np.split(y_test, n_chunks)
    return x_train, y_train, x_test, y_test


def knn_predictions(x_train, y_train, x_test, k_neighbours):
    """Creates kNN classifier, fits it to training data and classifies 
    the test data based of of the training samples.
    Params:
        x_train, y_train: np.ndarray
            Training dataset
        x_test: np.ndarray
            Testing dataset to be predicted
        k_neighbours: int
            Number of neighbours in the kNN classifier
    Returns: 
        y_pred: np.ndarray
            Predicted values for the test set'
        fit_time: float
            Time used to fit training data
    """
    neigh = KNeighborsClassifier(n_neighbors=k_neighbours, n_jobs=-1, metric='euclidean') # distance = Euclidean and n_jobs = 1 by default
    tstart = time.time()
    neigh.fit(x_train, y_train) 
    fit_time = time.time() - tstart
    print(fit_time)
    return neigh.predict(x_test), fit_time


def sort_predictions_knn(y_pred, y_test):
    """Finds the wrong and correct predictions in y_pred and returns 
    arrays with the corresponding index.
    Params:
        y_pred: np.ndarray
            The values predicted by the classifier for the test set
        y_test: np.ndarray
            The true values for the test set
    Returns:
        failed_indexes: list
            indexes where the prediction was incorrect
        correct_indexes: list
            indexes where the prediction was correct
    """
    zipped = zip(y_pred, y_test)
    failed_indexes = []
    correct_indexes = []
    for i, (j, k) in enumerate(zipped):
        failed_indexes.append(i) if (j != k) else correct_indexes.append(i)
    return failed_indexes, correct_indexes


def display_predictions(failed_indexes, correct_indexes, x_test, y_test, n):
    """Displays the first n correct or incorrect predictions that the 
       classifier made.
    Params:
        failed_indexes: list
            indexes where the prediction was incorrect
        correct_indexes: list
            indexes where the prediction was correct
        x_test: np.ndarray
            The handwritten numbers from the testset
        y_test: np.ndarray
            The true values from the test set
        n: int
            Numbers of plots of correct or incorrect predictions
    """
    if failed_indexes:
        print(f"Displaying {n} misclassified pictures...")
        for i, failed in enumerate(failed_indexes):
            if i <= n:
                print(f"True number was: {y_test[failed]}\nPredicted number was: {y_pred[failed]}")
                plt.imshow(x_test[failed].reshape(28, 28))
                plt.show()
    elif correct_indexes:
        print(f"Displaying {n} correctly classified pictures...")
        for i, correct in enumerate(correct_indexes):
            if i <= n:
                print(f"Correctly classified number: {y_test[correct]}")
                plt.imshow(x_test[correct].reshape(28, 28))
                plt.show()
    else: return

def sort_data(data, labels):
    """Sorts all members of x_train into the classes from y_train.
    Params:
        data: np.ndarray
            Feature vectors 
        labels: np.ndarray
            Correct labels/values
    Retutns:
        data_sorted: dict
            Dict with sorted members. Key = class, 
            value = array of feature vectors corresponding to that class
    """
    n_classes = len(np.unique(labels)) 
    # Init dict as {0: [], 1: [], ...} with n_classes empty arrays as values
    data_sorted = {k: [] for k in range(n_classes)}
    # Fill dict with correct members from x_train
    for i in range(len(labels)):
        data_sorted[labels[i]].append(data[i])
    return data_sorted

def cluster(data, labels, M):
    """ Clusters each class into M clusters with Kmeans(), each cluster 
        center acts as a template vector for the given class in the returned dict 
    Params:
        data: np.ndarray
            Feature vectors 
        label: np.ndarray
            Correct labels/values
        M: int
            Number of clusters
    Returns:
        x_train_small: np.ndarray
            ...
        y_train_small: np.ndarray
            ...
    """
    data_sorted = sort_data(data, labels)
    # Vectors for extending the dict into 2D training vectors
    x_train_modified, y_train_modified = [], [] 
    for class_i, members in data_sorted.items():
        # Fit members to class with M clusters
        kmeans = KMeans(n_clusters=M).fit(members)
        # Insert array of cluster centers as new members to class i_class
        x_train_modified.extend(kmeans.cluster_centers_)
        # Append M elements of the class to make x and y have equal lengths
        y_train_modified.extend(np.repeat(class_i, M))

    # Convert to np.ndarray for passing to kNN classifier.
    # Because means (cluster centroids) have lots of decimals, we 
    # need .astype(int) to make arrays sparse, i.e. very small vals->0.
    # If not then clustering won't be more effective than "normal" kNN... 
    x_train_modified = np.array(x_train_modified).astype(int)
    y_train_modified = np.array(y_train_modified).astype(int)

    return x_train_modified, y_train_modified

# END utilities




if __name__ == '__main__':
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten data, split into n chuncks for time saving, n_chunks = 1 if whole dataset is to be used
    n_chuncks = 10
    x_train, y_train, x_test, y_test = flatten_split_data(x_train, y_train, x_test, y_test, n_chuncks)
    # Choose which data chunck to use, 0 is fine - Change it for slightly different results
    chunk = 3
    x_train, y_train, x_test, y_test = x_train[chunk], y_train[chunk], x_test[chunk], y_test[chunk]

    # # BEGIN Task 1A
    # # Classify the values in the test set with a kNN-classifier with k_neighs neighbours
    # k_neighs = 3
    # y_pred = knn_predictions(x_train, y_train, x_test, k_neighs)
    # # Display confusion matrix and error rate for the classifier
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()
    # print(classification_report(y_test, y_pred)) # Error rate is given in report
    # # END Task 1A

    # # BEGIN Task 1B
    # failed_indexes, correct_indexes = sort_predictions_knn(y_pred, y_test)
    # n_fails = 4
    # # Show some (n_fails) misclassified pictures
    # display_predictions(failed_indexes, [], x_test, y_test, n_fails)
    # # END Task 1B

    # # BEGIN Task 1C
    # n_corrects = 4
    # # Show some (n_corrects) correctly classified pictures
    # display_predictions([], correct_indexes, x_test, y_test, n_corrects)
    # # END Task 1C

    # BEGIN Task 2A
    d, x_train_clustr, y_train_clustr = cluster(x_train, y_train, M=64)
    # Prove correct shape = (64, 28x28)
    # for i in range(9):
    #     print(str(d[i].shape))
    # print(x)
    # print("x_train:", str(x_train.shape))
    # print("y_train:", str(y_train.shape))
    # print("x_train_c:", str(x_train_clustr.shape))
    # print("y_train_c:", str(y_train_clustr.shape))
    k_neighs = 3
    y_pred_clustr, t2 = knn_predictions(x_train_clustr, y_train_clustr, x_test, k_neighs)
    y_pred, t1 = knn_predictions(x_train, y_train, x_test, k_neighs)
    print(f'Fit times:\nWithout clustering: {t1}\nWith clustering: {t2}')
    for i in range(500, 600):
        print(x_train_clustr[i])
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print(classification_report(y_test, y_pred_clustr)) 
    # END Task 2A





