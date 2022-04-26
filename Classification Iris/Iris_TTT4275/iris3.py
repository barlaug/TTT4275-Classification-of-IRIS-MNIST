# Importing the modules
from distutils.log import error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, zero_one_loss
from sklearn.datasets import load_iris


def load_data():
    """Loads iris with sklearn
    Returns:
        dataset_iris: np.ndarray
            Features for all samples for every class
        legendslist_iris: np.ndarray 
            Array of target_labels (3,1) 
        target_vector: np.ndarray
            Vector containing 150 corresponding targets 
    """
    iris = load_iris()
    dataset_iris = np.array(iris.data)
    legendslist_iris = np.array(iris.target_names)
    target_vector = np.array(iris.target)

    return dataset_iris, legendslist_iris, target_vector


def extract_sets(data, targets, train_set_size, num_classes):
    """Splits data into training set and testing set
    Params:
       data: np.ndarray
            Full iris dataset, with all samples for every class
       train_set_size: int
            Desired size of training set
        num_classes: int
            Desired size of training set
    Returns:
        train_samples, test_samples: np.ndarray
        t_train, t_test: np.ndarray
            t vector of index for 
        y_train, y_test: np.array
            True outputs corresponding to the training set and 
                testset respectively
    """
    N = int(len(data)/num_classes)

    test_set_size = N - train_set_size

    train_samples = np.zeros((num_classes*train_set_size,int(len(data[0]))+1))
    test_samples = np.zeros((num_classes*test_set_size,int(len(data[0]))+1))
    t_train = np.zeros((num_classes*train_set_size, num_classes))
    t_test = np.zeros((num_classes*test_set_size, num_classes))
    y_train = np.zeros((num_classes*train_set_size))
    y_test = np.zeros((num_classes*test_set_size))

    t_vec = np.zeros((int(len(targets)), num_classes)) 
    #Produce targetmatrix corresponding to the right classes
    for i in range(int(len(targets))):
        t_vec[i][targets[i]] = 1

    for i in range(num_classes):
        feature_temp = data[(N*i):((i+1)*N)]
        target_temp = t_vec[(N*i):((i+1)*N)]
        y_temp = targets[(N*i):((i+1)*N)]

        # Add the ones in every x[4], for features
        feature_number = np.ones((feature_temp.shape[0],1))
        feature_temp = np.hstack((feature_temp, feature_number))

        train_samples[(train_set_size*i):((i+1)*train_set_size), :] = feature_temp[ :train_set_size,:]
        test_samples[(test_set_size*i):((i+1)*test_set_size), :] = feature_temp[train_set_size:,:]

        t_train[(train_set_size*i):((i+1)*train_set_size), :] = target_temp[ :train_set_size, :]
        t_test[(test_set_size*i):((i+1)*test_set_size), :] = target_temp[train_set_size:,:]

        y_train[(train_set_size*i):((i+1)*train_set_size)] = y_temp[ :train_set_size]
        y_test[(test_set_size*i):((i+1)*test_set_size)] = y_temp[train_set_size:]

    return train_samples, test_samples, t_train, t_test, y_train, y_test


def sigmoid(z_k):
    """Sigmoid function. Equation (20) in compendium
    Params:
       z_k: np.ndarray
    Returns:
        np.ndarray
    """
    return 1/(1+np.exp(-z_k))


def train_lin_model(train_samples, train_targets, alpha, num_iterations, num_classes):
    """Trains the linear model. Equations (19)-(23) in compendium implemented
    Params:
        train_samples: np.array
            The training samples
        train_targets: np.array
            The targets, corresponding to the training samples
        alpha: float
            learning rate
        num_iterations: int
            Number of iterations we wish to use
    Returns:
        W: np.array
            Cx(D+1) classifier
        MSEarray: np.array
            Array containing MSEs from the calculations. For plotting
    """
    num_features = int(len(train_samples[0]))
    W = np.zeros((num_classes,num_features))
    x_k = np.zeros((1, num_features))
    t_k = np.zeros((num_classes,1))
    z_k = np.zeros((num_classes,1))
    g_k = np.zeros((num_classes,1)) # The input values we want to minimise MSE for
    g_k[0] = 1 # Start with 1
    MSEset = []
    for i in range(num_iterations):
        MSE = 0
        grad_W_MSE = 0
        for k in range(int(len(train_samples))):
            x_k = np.reshape(train_samples[k], (1, num_features))
            z_k = np.reshape(np.matmul(W, x_k.T), (num_classes, 1))
            g_k = np.reshape(sigmoid(z_k), (num_classes, 1))
            t_k = np.reshape(train_targets[k], (num_classes, 1))

            grad_g_MSE = g_k - t_k
            grad_z_gk = np.multiply(g_k,(1 - g_k)) #element wise
            grad_W_z = x_k
            grad = np.multiply(grad_g_MSE,grad_z_gk)

            grad_W_MSE = grad_W_MSE + np.matmul(grad, grad_W_z)

            MSE = MSE + 0.5*np.multiply((g_k - t_k).T, (g_k - t_k))
        
        MSEset.append(MSE)
        W = W - alpha*grad_W_MSE
    MSEarray = np.array(MSEset)
    return W #, MSEarray


def discriminant_classifier(W, test_samples):
    """Decision rule and discriminant classifier. Equations (6)-(7) in compendium implemented
    Params:
        W: np.array
            Cx(D+1) classifier
        test_samples: np.ndarray

    Returns:
        y_pred: nparray
            Array with predicted classes denoted with indexes 0-2 
    """
    x_test = np.zeros((1, len(test_samples[0])))
    g_i = np.zeros((num_classes,1))
    y_pred = np.zeros(len(test_samples))    
    for i in range(len(test_samples)):
        x_test = test_samples[i] 
        g_i = np.matmul(W, x_test)
        g_j = int(np.argmax(g_i))
        y_pred[i] = g_j
    return y_pred


def tune_alpha(alphas, x, targets, train_set_size, iterations, num_classes):
    error_rate_set = []
    x_train, x_test, t_train, t_test, y_train, y_true = extract_sets(x, targets, train_set_size, num_classes)

    for alpha in alphas:
        #last_error_rate = 1
        W = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
        y_pred = discriminant_classifier(W, x_test)
        error_rate = get_error_rate(y_true, y_pred)
        #if error_rate < last_error_rate:
        #    best_alpha = alpha
        #last_error_rate = error_rate
        error_rate_set.append(error_rate)

    error_rate_vec = np.array(error_rate_set)
    alpha_vec = np.array(alphas)
    plt.scatter(alpha_vec, error_rate_vec)
    plt.ylabel("Error rate")
    plt.xlabel("alpha")
    plt.title("Error rate as a function of alpha")
    #print("Alpha value:", best_alpha)
    plt.show()


 

def display_CM_Error(y_pred, y_true, legends):
    """Displays confusion matrix, classification report/errors(optional) and error rate
       for deviation between predicted and true labels
    Params:
        y_pred, y_true: np.ndarray
            Predicted and true labels/classes
    """
    error_rate = (zero_one_loss(y_true, y_pred))*100
    print(f"Error rate: {error_rate}%")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=legends)
    disp.plot()
    print('Displaying confusion matrix...')
    plt.show()
    #print('Classification report:')
    #print(classification_report(y_true, y_pred)) # Accuracy is given in report
    #Print error_rate


def get_error_rate(y_true, y_pred):
    """Returns error rate
    Params:
        y_true, y_pred: np.array
            True and predicted labels/classes
    Returns:
        error_rate: float
            Error_rate in percentage
    """
    accuracy = zero_one_loss(y_true, y_pred)
    error_rate = (accuracy)*100
    return error_rate


def plot_histogram(x, num_classes):
    """Displays histogram for the four features from every class
    Params:
        x: np.ndarray
            Complete iris dataset
        num_classes: int
            Number of classes
    """
    features = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    N = int(len(x)/num_classes)
    for i in range(len(features)):
        n_bins = np.arange(0, 8,1/2)
        plt.subplot(2, 2, i+1)
        #colors={0:'r',1:'g',2:'b',3:'k'}
        plt.hist([x[:N,i], x[N:2*N,i], x[2*N:len(x),i]], bins = n_bins) #, color = colors, ec = colors)
        plt.xlabel(features[i])
        plt.ylabel("Count")
        plt.xlabel("%s [cm]" % features[i])
    print("Plotting histogram")
    plt.show()


def remove_feature(x, feature_index):
    return np.delete(x, feature_index, axis=1)


if __name__ == '__main__':
    num_classes = 3
    iris_data, iris_legends, iris_targets = load_data()
    
    iterations = 2000
    train_set_size = 30
    test_set_size = 20 #needed for task 1d)

    #TASK 1
    plot_histogram(iris_data, num_classes)
    #alphas = np.linspace(0.01, 0.1, 250)
    #tune_alpha(alphas, iris_data, iris_targets, train_set_size, iterations, num_classes)

    print("\n-------------------------------------------------------------\n")

    alpha = [0.006]
    
    x_train, x_test, t_train, t_test, y_train, y_true = extract_sets(
        iris_data, iris_targets, train_set_size, num_classes)
    print("Task 1a)-c)")
    print("30 training samples, 20 test samples from each class")
    W1 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W1, x_test)
    display_CM_Error(y_pred_test, y_true, iris_legends)
    
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W1, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends) #Dimensions!


    print("\n-------------------------------------------------------------\n")
    #TASK 1d)
    x_test, x_train, t_test, t_train, y_true, y_train = extract_sets(
        iris_data, iris_targets, test_set_size, num_classes)
    print("Task 1d)")
    print("30 last samples as training samples, 20 first samples as test samples")
    W2 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W2, x_test)
    display_CM_Error(y_pred_test, y_true, iris_legends)
    
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W2, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends)
    
    print("\n-------------------------------------------------------------\n")
    #TASK 2
    print("Task 2a)")
    print("Removing feature(s) with the most overlap:")
    #removing 1 feature. Removing index 1 (Sepal width)
    print("Removing Sepal width:")
    iris_data_1 = remove_feature(iris_data,1)
    x_train, x_test, t_train, t_test, y_train, y_true = extract_sets(
        iris_data_1, iris_targets, train_set_size, num_classes)

    W3 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W3, x_test)
    display_CM_Error(y_pred_test, y_true, iris_legends)
    
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W3, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends)
    
    #"""
    #removing 2 features. Removing index 1, 0 ():
    print("Removing Sepal length:")
    iris_data_2 = remove_feature(iris_data_1,0)
    x_train, x_test, t_train, t_test, y_train, y_true = extract_sets(
        iris_data_2, iris_targets, train_set_size, num_classes)

    W4 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W4, x_test)
    display_CM_Error(y_pred_test, y_true, iris_legends)
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W4, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends)

    #removing 3 features. Removing 3 (1) ():
    print("Removing Petal width:")
    iris_data_3 = remove_feature(iris_data_2,1) #Hvilken index blir riktig å fjerne her?
    x_train, x_test, t_train, t_test, y_train, y_true = extract_sets(
        iris_data_3, iris_targets, train_set_size, num_classes)

    W5 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W5, x_test)
    display_CM_Error(y_pred_test, y_true, iris_legends)
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W5, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends)
    #"""

    print("\n-------------------------------------------------------------\n")

        
        