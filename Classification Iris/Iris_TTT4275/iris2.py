# Importing the modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy


import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, zero_one_loss
from sklearn.datasets import load_iris


Classes = 3



def load_data():
    """Loads iris with sklearn
    
    Returns:
        Target
    """
    iris = load_iris()
    data = iris.data
    dataset_iris = np.array(data)
    #print(dataset_iris)
    legends = iris.target_names
    legendslist_iris = np.array(legends)

    return dataset_iris, legendslist_iris


def split_data(data, train_set_size, num_classes):
    """Splits data into training set and testing set
    Params:
       data: np.ndarray
            Full iris dataset
       train_set_size: int
            Desired size of training set
        num_classes: int
            Desired size of training set
    Returns:
        Training set, testing set: np.ndarray
    """
    N = int(len(data)/num_classes)

    test_set_size = N - train_set_size
    train_samples = np.zeros((num_classes*train_set_size,int(len(data[0]))+1))
    test_samples = np.zeros((num_classes*test_set_size,int(len(data[0]))+1))

    for i in range(num_classes):
        temp = data[(N*i):((i+1)*N)]
        # Add the ones in every x[4]
        class_number = np.ones((temp.shape[0],1))
        temp = np.hstack((temp, class_number))
        print(temp.shape)
        train_samples[(train_set_size*i):((i+1)*train_set_size), :] = temp[ :train_set_size,:]
        test_samples[(test_set_size*i):((i+1)*test_set_size), :] = temp[train_set_size:,:]
        

    return train_samples, test_samples


def calculate_W(train_samples, test_samples, alpha, num_iterations):
    """calculates W. A CxD+1 classifier
    Params:
       train_samples: np.ndarray
            Training set
       test_samples: int
            Testing set
        features: int
            features ??
        alpha: int 
        num_iterations: int 
            Number of iterations 
    Returns:
        W: np.ndarray
            CxD+1 classifier
    """
    num_classes = Classes
    num_features = int(len(train_samples[0])) #4
    print(num_features)

    W = np.zeros((num_classes,num_features))
    for i in range(num_iterations):
        W = W - alpha*calculate_gradient_W_MSE(W, test_samples, train_samples).T
    return W


def calculate_gradient_W_MSE(W, t, x):
    """calculates gradient of MSE. Equation (22) in compendium
    Params:
       W: np.ndarray
            CxD+1 classifier
       t: np.ndarray
            testing samples
        x: np.ndarray
            training samples
    Returns:
        grad W: np.ndarray
    """
    print(x)
    z = np.dot(W, x.T)
    print(z)
    g = sigmoid(z)
    print(g)
    grad_g_MSE = g.T - t.T #.T?
    grad_z_g = np.multiply(g,(1 - g)) #element wise
    grad_W_z = np.dot(np.multiply(grad_g_MSE,grad_z_g), x.T) #np.transpose(xk)?
    return grad_W_z
 

def sigmoid(z):
    """Sigmoid function. Equation (20) in compendium
    Params:
       z
    Returns:
        grad W: np.ndarray
    """
    return 1/(1+np.exp(-z))


def calculate_MSE(W, t, x):
    """calculates MSE. Equation (19) in compendium
    Params:
       W: np.ndarray
            CxD+1 classifier
       t: np.ndarray
            testing samples
        x: np.ndarray
            training samples
    Returns:
        MSE: 
    """
    g =  np.dot(W, x.T)
    sum = 0
    for i in range(len(g)):
        error = np.subtract(g[i], t[i])
        error_T = np.transpose(error)
        sum += np.dot(np.matmul(error_T,error))
    return sum/2


def get_prediction(W, x):
    """Returns prediction matrix
    Params:
        W: np.array
            CxD+1 classifier
        x: np.array
            samples
    Returns:
        prediction_vec: np.array
            vector of predictions
    """
    z = W*x
    g = sigmoid(z)

    prediction_vec = np.zeros((len(x[0]), 1))
    for i in range(len(g[0])):
        prediction_vec[i] = np.argmax(g[:,i], axis=0)
    print('Predictions:')
    print(prediction_vec)
    return prediction_vec


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

    #accuracy = zero_one_score(y_test, y_pred)
    #error_rate = 1 - accuracy
    #print(f"Error rate: {error_rate}%")


if __name__ == '__main__':
    num_classes = 3
    iris_data, iris_legends = load_data()
    #alphas = [0.5, 0.25, 0.1, 0.05, 0.005]
    alphas = [0.05]
    iterations = 5000
    train_set_size = 30
    train_set, test_set = split_data(iris_data, train_set_size, num_classes)
    for alpha in alphas:
        print("30 train, 20 test")
        W = calculate_W(train_set[:,:], test_set, alpha, iterations) #[:,:]? 
        pred_train = get_prediction(W, train_set)
        pred_test =  get_prediction(W, test_set)
        print("Training")
        get_prediction(W, test_set)
        #display_CM_Error(y_pred, y_true)
        