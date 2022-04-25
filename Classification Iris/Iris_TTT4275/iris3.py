# Importing the modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy


#import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, zero_one_loss
from sklearn.datasets import load_iris


Classes = 3



def load_data():
    """Loads iris with sklearn
    
    Returns:
        Target
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
    y_true = np.zeros((num_classes*test_set_size))
    t_train = np.zeros((num_classes*train_set_size, num_classes))
    t_vec = np.zeros((int(len(targets)), num_classes)) 
    

    for i in range(int(len(targets))):
        t_vec[i][targets[i]] = 1

    for i in range(num_classes):
        class_temp = data[(N*i):((i+1)*N)]
        target_temp = t_vec[(N*i):((i+1)*N)]
        y_temp = targets[(N*i):((i+1)*N)]

        # Add the ones in every x[4]
        class_number = np.ones((class_temp.shape[0],1))
        class_temp = np.hstack((class_temp, class_number))

        train_samples[(train_set_size*i):((i+1)*train_set_size), :] = class_temp[ :train_set_size,:]
        test_samples[(test_set_size*i):((i+1)*test_set_size), :] = class_temp[train_set_size:,:]
        t_train[(train_set_size*i):((i+1)*train_set_size), :] = target_temp[ :train_set_size, :]
        y_true[(test_set_size*i):((i+1)*test_set_size)] = y_temp[train_set_size:]

    return train_samples, test_samples, t_train, y_true


def sigmoid(z_k):
    """Sigmoid function. Equation (20) in compendium
    Params:
       z
    Returns:
        grad W: np.ndarray
    """
    return 1/(1+np.exp(-z_k))


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


def train_lin_model(train_samples, train_targets, alpha, num_iterations):
    """Trains the linear model. Equations (19)-(23) in compendium implemented
    Params:
        train_samples: np.array
            The training samples
        alpha: float
            learning rate
        num_iterations: int
            Number of iterations we want to use
    Returns:
        W: np.array
            CxD+1 classifier
        MSEarray: np.array
            Array containing MSEs from the calculations. For ploting
    """
    num_classes = Classes
    num_features = int(len(train_samples[0])) #5

    W = np.zeros((num_classes,num_features))

    t_k = np.zeros((num_classes,1))
    g_k = np.zeros((num_classes,1)) # The input values we want to minimise MSE for
    g_k[0] = 1 # always start with 1
    MSEset = []
    for i in range(num_iterations):
        MSE = 0
        grad_W_MSE = 0
        for k in range(int(len(train_samples))):
            x_k = np.reshape(train_samples[k], (1, 5))
            z_k = np.matmul(W, x_k.T)
            g_k = np.reshape(sigmoid(z_k), (1, 3))


            t_k = np.reshape(train_targets[k], (1, 3))

            grad_g_MSE = g_k - t_k
            grad_z_gk = np.multiply(g_k,(1 - g_k)) #element wise
            grad_W_z = x_k

            grad = np.multiply(grad_g_MSE,grad_z_gk).T
            
            grad_W_MSE = grad_W_MSE + np.matmul(grad, grad_W_z)

            MSE = 0.5*np.multiply((g_k - t_k).T, (g_k - t_k))
        
        MSEset.append(MSE)
        W = W - alpha*grad_W_MSE
    MSEarray = np.array(MSEset)
    return W #, MSEarray

def discriminant_classifier(W, test_samples, y_true):
    """Decision rule and discriminant classifier. Equations (6)-(7) in compendium implemented
    Params:
        W: np.array
            CxD+1 classifier
        test_samples: np.ndarray

    Returns:
        g: 

    """
    x_test = np.zeros((1, len(test_samples[0])))
    g_k = np.zeros((num_classes,1))
    y_pred = np.zeros(len(test_samples))
    
    for i in range(len(test_samples)):
        x_test = np.reshape(test_samples[i,:], (1, 5))
        g_i = np.matmul(W, x_test.T)
        print(g_i)
        g_j = int(np.argmax(g_i))
        print(g_j)
        y_pred[i] = g_j
    print(y_pred)


    return y_pred






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

def print_error_rate(y_true, y_pred):
    accuracy = zero_one_loss(y_true, y_pred)
    error_rate = (1 - accuracy)*100
    print(f"Error rate: {error_rate}%")


if __name__ == '__main__':
    num_classes = 3
    iris_data, iris_legends, iris_targets = load_data()
    #alphas = [0.5, 0.25, 0.1, 0.05, 0.005]
    alphas = [0.05]
    iterations = 1000
    train_set_size = 30
    train_set, test_set, t_train, y_true = extract_sets(iris_data, iris_targets, train_set_size, num_classes)
    for alpha in alphas:
        print("30 train, 20 test")
        W = train_lin_model(train_set[:,:], t_train, alpha, iterations) #[:,:]? 
        #pred_train = get_prediction(W, train_set)
        #pred_test =  get_prediction(W, test_set)
        print("Training")
        print(W)
        #get_prediction(W, test_set)
        y_pred = discriminant_classifier(W, test_set, y_true)
        display_CM_Error(y_pred, y_true)
        print_error_rate(y_true, y_pred)
        
        