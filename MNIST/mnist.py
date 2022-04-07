import numpy as np
import time
import random
from matplotlib import pyplot as plt
from keras.datasets import mnist #!NB: pip install tensorflow, pip install keras -> Skriv i readme.
from sklearn.neighbors import KNeighborsClassifier # NB: pip install sklearn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans

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
    # Flatten 2D images into 1D arrays
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


def knn(x_train, y_train, x_test, k_neighs):
    """Creates kNN classifier, fits it to training data and classifies the
       test data based of of the training samples. Also times fitting and prediction.
    Params:
        x_train, y_train: np.ndarray
            Training dataset
        x_test: np.ndarray
            Testing dataset to be predicted
        k_neighbours: int
            Number of neighbours to be used in the kNN classifier
    Returns: 
        y_pred: np.ndarray
            Predicted values for the test set
        fit_time: float
            Time used to fit training data
        pred_time: float
            Time used to classify/predict the test data
    """
    neigh = KNeighborsClassifier(n_neighbors=k_neighs, metric='euclidean')
    # Fit and predict. Time both operations
    t0_fit = time.time()
    neigh.fit(x_train, y_train) 
    fit_time = time.time() - t0_fit
    t0_pred = time.time()
    y_pred = neigh.predict(x_test)
    pred_time = time.time() - t0_pred

    return y_pred, fit_time, pred_time


def sort_predictions_knn(y_pred, y_test):
    """Finds the true and false predictions in y_pred and returns 
    arrays with the corresponding index.
    Params:
        y_pred: np.ndarray
            Predicted labels from some test set
        y_test: np.ndarray
            True labels for the same set
    Returns:
        failed_indexes: list
            Indexes with incorrect predictions
        correct_indexes: list
            Indexes with correct predictions
    """
    zipped = zip(y_pred, y_test)
    failed_indexes = []
    correct_indexes = []
    for i, (j, k) in enumerate(zipped):
        failed_indexes.append(i) if j != k else correct_indexes.append(i)
    
    return failed_indexes, correct_indexes


def display_predictions(failed_indexes, correct_indexes, x_test, y_test, y_pred, n):
    """Displays n random samples from the correct or incorrect predictions that 
       the classifier made and display them.
    Params:
        failed_indexes: list
            Indexes with incorrect predictions
        correct_indexes: list
            Indexes with correct predictions
        x_test: np.ndarray
            Handwritten numbers from the test set
        y_test: np.ndarray
            True values for the test set
        y_pred: np.ndarray
            Predicted values
        n: int
            Numbers of examples to show
    """
    if failed_indexes:
        # Get n random samples from failed idx list
        rand_failsample = random.sample(failed_indexes, n)
        print(f"Displaying {n} misclassified pictures...")
        for failed_idx in rand_failsample:
            print(f"True number was: {y_test[failed_idx]}\nPredicted number was: {y_pred[failed_idx]}\n")
            plt.imshow(x_test[failed_idx].reshape(28, 28))
            plt.show()
    if correct_indexes:
        # Get n random samples from correct idx list
        rand_corrsample = random.sample(correct_indexes, n)
        print(f"Displaying {n} correctly classified pictures...")
        for corr_idx in rand_corrsample:
            print(f"True number was: {y_test[corr_idx]}\nPredicted number was: {y_pred[corr_idx]}\n")
            plt.imshow(x_test[corr_idx].reshape(28, 28))
            plt.show()
    else: return

def sort_data(data, labels):
    """Sorts all members of data into the classes from labels.
    Params:
        data: np.ndarray
            Feature vectors, i.e. x_train
        labels: np.ndarray
            Correct labels/values, i.e. y_train
    Returns:
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
    """ Clusters each class into M clusters using the Kmeans algorithm. 
        Each cluster mean acts as a template vector for the given class 
        in the returned condensed training dataset. 
    Params:
        data: np.ndarray
            Feature vectors 
        label: np.ndarray
            Correct labels/values
        M: int
            Number of clusters to be used in Kmeans
    Returns:
        x_train_small: np.ndarray
            Array of new template vectors
        y_train_small: np.ndarray
            True labels/classes for the template vectors
    """
    data_sorted = sort_data(data, labels)
    # Cointainers for extending the data_sorted dict into 2D training vectors
    x_train_small, y_train_small = [], [] 
    for class_i, members in data_sorted.items():
        # Compute kmeans clustering for given class
        kmeans = KMeans(n_clusters=M).fit(members)
        # Insert array of cluster centroids/means as new members to class i_class
        x_train_small.extend(kmeans.cluster_centers_)
        # Append M elements of the class to new y_train to make x and y have equal lengths
        y_train_small.extend(np.repeat(class_i, M))

    # Convert to integer np.ndarrays for passing to kNN classifier
    x_train_small = np.array(x_train_small).astype(int)
    y_train_small = np.array(y_train_small).astype(int)

    return x_train_small, y_train_small

def display_CM_Error(y_pred, y_true):
    """Displays confusion matrix and classification report/errors
       for deviation between predicted and true labels
    Params:
        y_pred, y_true: np.ndarray
            Predicted and true labels/classes
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    print('Displaying confusion matrix...')
    plt.show()
    print('Classification report:')
    print(classification_report(y_true, y_pred)) # Accuracy is given in report


def k_performancemeasure(k_max, x_train, y_train, x_test, y_test):
    """Plots mean error, fit time and prediction time as functions 
       of K (number of neighbours) in the kNN classsifier.
    """
    fit_times = []
    pred_times = []
    error = []
    for k in range(1, k_max):
        y_pred, tk_fit, tk_pred = knn(x_train, y_train, x_test, k)
        fit_times.append(tk_fit)
        pred_times.append(tk_pred)
        error.append(np.mean(y_pred != y_test))
        
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle('Performance with different K-values')
    ax[0].plot(range(1, k_max), error, color='black', linestyle='dashed', marker='o',
            markerfacecolor='red')
    ax[0].set_ylabel('Mean Error')

    ax[1].plot(range(1, k_max), fit_times, color='black', linestyle='dashed', marker='o',
            markerfacecolor='red')
    ax[1].set_ylabel('Fit Time [s]')

    ax[2].plot(range(1, k_max), pred_times, color='black', linestyle='dashed', marker='o',
            markerfacecolor='red')
    ax[2].set_xlabel('K Value')
    ax[2].set_ylabel('Prediction Time [s]')
    plt.show()
    
# END utilities


if __name__ == '__main__':
    ######### PRESENT TASKS ##########

    # """
    # TASK 1
    # (For this task you may comment out entire TASK 2)

    # GLOBALS: 
    # Change them as you like, recommended n_chunks = 5
    n_chunks = 5
    # Choose which data chunck to use after splitted, 0 is fine - Change it for slightly different results
    chunk = 0
    # Choose number of falsely predicted images to display in Task 1B
    n_fails = 4
    # Choose number of correctly predicted images to display in Task 1C
    n_corrects = 5

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Change to np.array and type to int/float
    x_train = np.asarray(x_train).astype(float)
    y_train = np.asarray(y_train).astype(int)
    x_test = np.asarray(x_test).astype(float)
    y_test = np.asarray(y_test).astype(int)
    # Flatten data, split into n chuncks for time saving, n_chunks = 1 if whole dataset is to be used
    x_train, y_train, x_test, y_test = flatten_split_data(x_train, y_train, x_test, y_test, n_chunks)
    x_train, y_train, x_test, y_test = x_train[chunk], y_train[chunk], x_test[chunk], y_test[chunk]

    # Classify the values in the test set with a kNN-classifier with k_neighs neighbours
    y_pred, _, _ = knn(x_train, y_train, x_test, k_neighs=3)

    # Display confusion matrix and error rate for the classifier
    display_CM_Error(y_pred, y_test)

    # Sort failed and correctly predicted pictures by index 
    failed_indexes, correct_indexes = sort_predictions_knn(y_pred, y_test)
    # Show some (n_fails) misclassified pictures
    display_predictions(failed_indexes, [], x_test, y_test, y_pred, n_fails)
    # Show some (n_corrects) correctly classified pictures
    display_predictions([], correct_indexes, x_test, y_test, y_pred, n_corrects)

    # """ 
    # TASK 2
    # (For this task you may comment out entire TASK 1)
    
    # For task 2 we need the whole dataset; update globals and data vectors
    n_chunks = 1
    chunk = 0
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Change to np.array and type to int/float
    x_train = np.asarray(x_train).astype(float)
    y_train = np.asarray(y_train).astype(int)
    x_test = np.asarray(x_test).astype(float)
    y_test = np.asarray(y_test).astype(int)
    x_train, y_train, x_test, y_test = flatten_split_data(x_train, y_train, x_test, y_test, n_chunks)
    x_train, y_train, x_test, y_test = x_train[chunk], y_train[chunk], x_test[chunk], y_test[chunk]
    
    # Cluster the (whole) training dataset with classwise Kmeans clustering
    x_train_clustr, y_train_clustr = cluster(x_train, y_train, M=64)

    # Find confusion matrix and error rate for the kNN classifier from Task 1 using the M = 64 templates
    # per class with the whole dataset. Measure and display time spent on fitting and predicting 
    y_pred, tfit1, tpred1 = knn(x_train, y_train, x_test, k_neighs=3)
    y_pred_clustr, tfit2, tpred2 = knn(x_train_clustr, y_train_clustr, x_test, k_neighs=3)
    print('K = 3:')
    print("Confusion matrix and error without clustering:")
    display_CM_Error(y_pred, y_test)
    print("Confusion matrix and error with clustering:")
    display_CM_Error(y_pred_clustr, y_test)
    print(f'Fit times for training set of length {int(len(x_train)/n_chunks)}:\n\tWithout clustering: {tfit1}s\n\tWith clustering: {tfit2}s')
    print(f'\nPrediction times for test set of length {int(len(x_test)/n_chunks)}:\n\tWithout clustering: {tpred1}s\n\tWith clustering: {tpred2}s')
    
    # Design kNN classifier with k = 7. Find confusion matrix and error rate. Compare with previous systems
    y_pred_k7, tfit1_k7 , tpred1_k7 = knn(x_train, y_train, x_test, k_neighs=7)
    y_pred_clustr_k7, tfit2_k7, tpred2_k7 = knn(x_train_clustr, y_train_clustr, x_test, k_neighs=7)
    print('K = 7:')
    print("Confusion matrix and error without clustering:")
    display_CM_Error(y_pred_k7, y_test)
    print("Confusion matrix and error with clustering:")
    display_CM_Error(y_pred_clustr_k7, y_test)
    print(f'Fit times for training set of length {int(len(x_train)/n_chunks)}:\n\tWithout clustering: {tfit1_k7}s\n\tWith clustering: {tfit2_k7}s')
    print(f'\nPrediction times for test set of length {int(len(x_test)/n_chunks)}:\n\tWithout clustering: {tpred1_k7}s\n\tWith clustering: {tpred2_k7}s')

    # """

    # """
    # Extra: 
    # (For this part, you may comment out the rest of main)
    # Plot kNN performance as a function of number of neighbours k
    # Takes a while to compute, as it evaluates knn() k_max times...
    k_max = 30
    k_performancemeasure(k_max, x_train, y_train, x_test, y_test)

    # """







