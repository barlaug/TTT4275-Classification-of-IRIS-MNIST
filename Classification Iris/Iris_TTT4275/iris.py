# Importing the modules
from distutils.log import error
import numpy as np

from sklearn.metrics import zero_one_loss
from sklearn.datasets import load_iris

import plot

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
        targets: np.ndarray
            Array of target
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
    """Trains the linear model. Equations (19) and (21)-(23) in compendium implemented
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
        W_set, iter_set, MSE_set: array
            Arrays containing samples of W, iteration number and MSE, respectively. For plotting
    """
    num_features = int(len(train_samples[0]))
    W = np.zeros((num_classes,num_features))
    g_k = np.zeros((num_classes,1)) # The input values we want to minimise MSE for
    g_k[0] = 1 # Start with 1
    MSE_set = []
    W_set = []
    iter_set = []
    for i in range(num_iterations):
        MSE = 0
        grad_W_MSE = 0
        for k in range(int(len(train_samples))):
            x_k = np.reshape(train_samples[k], (1, num_features))
            z_k = np.reshape(np.matmul(W, x_k.T), (num_classes, 1))
            g_k = np.reshape(sigmoid(z_k), (num_classes, 1))
            t_k = np.reshape(train_targets[k], (num_classes, 1))

            grad_g_MSE = g_k - t_k
            grad_z_gk = np.multiply(g_k,(1 - g_k))
            grad_W_z = x_k
            grad = np.multiply(grad_g_MSE,grad_z_gk)

            grad_W_MSE = grad_W_MSE + np.matmul(grad, grad_W_z)

            MSE = MSE + 0.5*np.multiply((g_k - t_k).T, (g_k - t_k))
        W = W - alpha*grad_W_MSE
        #Sampling every 50 iteration for plotting
        #Comment this in if you want to plot Error rate vs iterations 
        """
        if not i % 20:
            W_set.append(W)
            MSE_set.append(MSE)
            iter_set.append(i)
    return W, W_set, iter_set, MSE_set"""
    return W


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


def get_error_rate(y_true, y_pred):
    """Returns error rate
    Params:
        y_true, y_pred: np.array
            True and predicted labels/classes
    Returns:
        error_rate: float
            Error_rate in percentage
    """
    error_rate = (zero_one_loss(y_true, y_pred))*100
    return error_rate


def remove_feature(x, feature_index):
    """Removes feature
    Params:
        x: np.ndarray
            Complete iris dataset
        feature_index: int
            index of feature to remove
    Returns:
        modified dataset: np.ndarray
    """
    return np.delete(x, feature_index, axis=1)


if __name__ == '__main__':
    num_classes = 3
    iris_data, iris_legends, iris_targets = load_data()
    iterations = 2000
    train_set_size = 30
    test_set_size = 20 #needed for task 1d)

    #plot.plot_histogram(iris_data, num_classes, iris_legends)
    #plot.display_ER_MSE_iterations(iris_data, iris_targets, train_set_size, num_classes)

    #TASK 1
    print("\n-------------------------------------------------------------\n")
    alpha = 0.006
    x_train, x_test, t_train, t_test, y_train, y_test = extract_sets(
        iris_data, iris_targets, train_set_size, num_classes)
    print("Task 1a)-c)\n 30 training samples, 20 test samples from each class")
    W1 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W1, x_test)
    plot.display_CM_Error(y_pred_test, y_test, iris_legends)
    
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W1, x_train)
    plot.display_CM_Error(y_pred_train, y_train, iris_legends)

    print("\n-------------------------------------------------------------\n")
    #TASK 1d)
    x_test, x_train, t_test, t_train, y_test, y_train = extract_sets(
        iris_data, iris_targets, test_set_size, num_classes)
    print("Task 1d)\n 30 last samples as training samples, 20 first samples as test samples")
    W2 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W2, x_test)
    plot.display_CM_Error(y_pred_test, y_test, iris_legends)
    
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W2, x_train)
    plot.display_CM_Error(y_pred_train, y_train, iris_legends)
    
    print("\n-------------------------------------------------------------\n")
    #TASK 2
    print("Task 2a)\n Removing feature(s) with the most overlap:")
    print("Removing Sepal width:")
    iris_data_1 = remove_feature(iris_data,1)
    x_train, x_test, t_train, t_test, y_train, y_test = extract_sets(
        iris_data_1, iris_targets, train_set_size, num_classes)

    W3 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W3, x_test)
    plot.display_CM_Error(y_pred_test, y_test, iris_legends)
    
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W3, x_train)
    plot.display_CM_Error(y_pred_train, y_train, iris_legends)
    #"""
    print("Removing Sepal length:")
    iris_data_2 = remove_feature(iris_data_1,0)
    x_train, x_test, t_train, t_test, y_train, y_test = extract_sets(
        iris_data_2, iris_targets, train_set_size, num_classes)

    W4 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W4, x_test)
    plot.display_CM_Error(y_pred_test, y_test, iris_legends)
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W4, x_train)
    plot.display_CM_Error(y_pred_train, y_train, iris_legends)

    print("Removing Petal width:")
    iris_data_3 = remove_feature(iris_data_2,1) #Hvilken index blir riktig å fjerne her?
    x_train, x_test, t_train, t_test, y_train, y_test = extract_sets(
        iris_data_3, iris_targets, train_set_size, num_classes)

    W5 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W5, x_test)
    plot.display_CM_Error(y_pred_test, y_test, iris_legends)
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W5, x_train)
    plot.display_CM_Error(y_pred_train, y_train, iris_legends)
    #"""        