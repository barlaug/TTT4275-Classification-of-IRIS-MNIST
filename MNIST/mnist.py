import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist #!NB: pip install tensorflow, pip install keras -> Skriv i readme.
from sklearn.neighbors import KNeighborsClassifier # NB: pip install sklearn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# TODO: 
# Add description of file structure and what is does
# se på performance for ulike k-verdier, se: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/


# BEGIN Utilities for Task 1

def prepare_data(x_train, y_train, x_test, y_test, n_chunks):
    """Reshapes datasets to 2D-arrays and splits them into n chunks

    Params:
        x_train, y_train: Training dataset
        x_test, y_test: Testing dataset
        n_chunks: Number of chunks to split data into
    Returns:
        Reshaped arrays
    """
    nsamples_train, nx_train, ny_train = x_train.shape
    x_train = x_train.reshape((nsamples_train,nx_train*ny_train))
    nsamples_test, nx_test, ny_test = x_test.shape
    x_test = x_test.reshape((nsamples_test,nx_test*ny_test))
    x_train = np.split(x_train, n_chunks)
    y_train = np.split(y_train, n_chunks)
    x_test = np.split(x_test, n_chunks)
    y_test = np.split(y_test, n_chunks)
    return x_train, y_train, x_test, y_test


def knn_predictions(x_train, y_train, x_test, k_neighbours):
    """
    Creates kNN classifier, fits it to training data and 
    classifies the test data based of of the training samples

    Params:
        train_x, train_y: Training dataset
        x_test: Testing dataset to be predicted
        k_neighbours: Number of neighbours in the kNN classifier
    Returns: 
        y_pred: Predicted values for the test set
    """
    neigh = KNeighborsClassifier(n_neighbors=k_neighbours) # distance = Euclidean and n_jobs = 1 by default
    neigh.fit(x_train, y_train) 
    return neigh.predict(x_test)


def sort_predictions(y_pred, y_test):
    """
    Finds the wrong and correct predictions in y_pred and
    returns arrays with the corresponding index

    Params:
        y_pred: The values predicted by the classifier for the test set
        y_test: The true values for the test set
    Returns:
        failed_indexes: Array containing indexes where the prediction was incorrect
        correct_indexes: Array containing indexes where the prediction was correct
    """
    zipped = zip(y_pred, y_test)
    failed_indexes = []
    correct_indexes = []
    for i, (j, k) in enumerate(zipped):
        #print(f"{i}: {j}, {k}" )
        failed_indexes.append(i) if (j != k) else correct_indexes.append(i)
    return failed_indexes, correct_indexes


def display_predictions(failed_indexes, correct_indexes, x_test, y_test, n):
    """
    Displays the first n correct or incorrect predictions that the 
    classifier made

    Params:
        failed_indexes: Array containing indexes where the prediction was incorrect
        correct_indexes: Array containing indexes where the prediction was correct
        x_test: The handwritten numbers from the testset
        y_test: The true values from the test set
        n: Numbers of plots of correct or incorrect predictions
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

# END utilities for Task 1


if __name__ == '__main__':
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Prepare data, split into n chuncks for time saving, n_chunks = 1 if whole dataset is to be used
    n_chuncks = 10
    x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test, n_chuncks)
    # Choose which data chunck to use, 0 is fine - Change it for slightly different results
    chunk = 0
    x_train, y_train, x_test, y_test = x_train[chunk], y_train[chunk], x_test[chunk], y_test[chunk]

    # BEGIN Task 1A
    # Classify the values in the test set with a kNN-classifier with k_neighs neighbours
    k_neighs = 3
    y_pred = knn_predictions(x_train, y_train, x_test, k_neighs)
    # Display confusion matrix and error rate for the classifier
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print(classification_report(y_test, y_pred)) # Error rate is given in report
    # END Task 1A

    # # BEGIN Task 1B
    failed_indexes, correct_indexes = sort_predictions(y_pred, y_test)
    # n_fails = 4
    # # Show some (n_fails) misclassified pictures
    # display_predictions(failed_indexes, [], x_test, y_test, n_fails)
    # # END Task 1B

    # BEGIN Task 1C
    n_corrects = 4
    # Show some (n_corrects) correctly classified pictures
    display_predictions([], correct_indexes, x_test, y_test, n_corrects)
    # END Task 1C





