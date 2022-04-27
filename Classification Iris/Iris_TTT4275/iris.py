# Importing the modules
from distutils.log import error
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, zero_one_loss
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
            grad_z_gk = np.multiply(g_k,(1 - g_k)) #element wise
            grad_W_z = x_k
            grad = np.multiply(grad_g_MSE,grad_z_gk)

            grad_W_MSE = grad_W_MSE + np.matmul(grad, grad_W_z)

            MSE = MSE + 0.5*np.multiply((g_k - t_k).T, (g_k - t_k))
        W = W - alpha*grad_W_MSE
        #Sampling every 50 iteration for plotting
        """Comment this in if you want to plot Error rate vs iterations 
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


def display_ER_MSE_iterations(x, targets, train_set_size, num_classes):
    x_train, x_test, t_train, t_test, y_train, y_test = extract_sets(x, targets, train_set_size, num_classes)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    #Values for which we want to test
    alphas = [0.0005, 0.001, 0.005, 0.05]#, 0.001, 0.005, 0.01, 0.05]
    iterations = 3000
    for alpha in alphas:
        print("Alpha: " + str(alpha))
        error_rate_set = []
        alpha_label = "alpha="+ str(alpha)
        W1, W_set, iter_set, MSE_set = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
        for W in W_set:
            y_pred = discriminant_classifier(W, x_test)
            error_rate = get_error_rate(y_test, y_pred)
            error_rate_set.append(error_rate)

        error_rate_vec = np.array(error_rate_set)
        iteration_vec = np.array(iter_set)
        MSE_vec = np.array(MSE_set)
        ax1.plot(iteration_vec, error_rate_vec, label=alpha_label)
        ax1.legend(fontsize=12)
        ax2.plot(iteration_vec, MSE_vec[:,0,0], label=alpha_label)
        ax2.legend(fontsize=12)
    ax1.set_ylabel("Error rate", fontsize=14)
    ax2.set_ylabel("MSE", fontsize=14)
    #ax1.set_xlabel("Iterations", fontsize=14)
    ax2.set_xlabel("Iterations", fontsize=14)
    ax1.set_title("Error rate vs Iterations for different alpha", fontsize=16)
    ax2.set_title("MSE vs Iterations for different alpha", fontsize=16)
    
    print('Plotting ER vs. iterations...')
    plt.savefig('error_rate_MSE_vs_iterations2.eps', format='eps')
    plt.show()


def display_CM_Error(y_pred, y_true, legends):
    """Displays confusion matrix and error rate
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


def plot_histogram(x, num_classes, labels):
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
        plt.hist(x[:N,i], alpha=0.5, label=labels[0])
        plt.hist(x[N:2*N,i], alpha=0.5, label=labels[0])
        plt.hist(x[2*N:len(x),i], alpha=0.5, label=labels[0])
        plt.legend()
        plt.xlabel(features[i])
        plt.ylabel("Count")
        plt.xlabel("%s [cm]" % features[i])
    #plt.savefig('histogram.eps', format='eps')
    print("Plotting histogram")
    plt.show()


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

    #plot_histogram(iris_data, num_classes, iris_legends)
    #display_ER_MSE_iterations(iris_data, iris_targets, train_set_size, num_classes)

    #TASK 1
    print("\n-------------------------------------------------------------\n")
    alpha = 0.006
    x_train, x_test, t_train, t_test, y_train, y_test = extract_sets(
        iris_data, iris_targets, train_set_size, num_classes)
    print("Task 1a)-c)")
    print("30 training samples, 20 test samples from each class")
    W1 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W1, x_test)
    display_CM_Error(y_pred_test, y_test, iris_legends)
    
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W1, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends)


    print("\n-------------------------------------------------------------\n")
    #TASK 1d)
    x_test, x_train, t_test, t_train, y_test, y_train = extract_sets(
        iris_data, iris_targets, test_set_size, num_classes)
    print("Task 1d)")
    print("30 last samples as training samples, 20 first samples as test samples")
    W2 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W2, x_test)
    display_CM_Error(y_pred_test, y_test, iris_legends)
    
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W2, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends)
    
    print("\n-------------------------------------------------------------\n")
    #TASK 2
    print("Task 2a)")
    print("Removing feature(s) with the most overlap:")
    print("Removing Sepal width:")
    iris_data_1 = remove_feature(iris_data,1)
    x_train, x_test, t_train, t_test, y_train, y_test = extract_sets(
        iris_data_1, iris_targets, train_set_size, num_classes)

    W3 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W3, x_test)
    display_CM_Error(y_pred_test, y_test, iris_legends)
    
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W3, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends)
    
    #"""
    print("Removing Sepal length:")
    iris_data_2 = remove_feature(iris_data_1,0)
    x_train, x_test, t_train, t_test, y_train, y_test = extract_sets(
        iris_data_2, iris_targets, train_set_size, num_classes)

    W4 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W4, x_test)
    display_CM_Error(y_pred_test, y_test, iris_legends)
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W4, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends)

    print("Removing Petal width:")
    iris_data_3 = remove_feature(iris_data_2,1) #Hvilken index blir riktig å fjerne her?
    x_train, x_test, t_train, t_test, y_train, y_test = extract_sets(
        iris_data_3, iris_targets, train_set_size, num_classes)

    W5 = train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
    print("Confusion matrix and error rate for the testing set:")
    y_pred_test = discriminant_classifier(W5, x_test)
    display_CM_Error(y_pred_test, y_test, iris_legends)
    print("Confusion matrix and error rate for the training set:")
    y_pred_train = discriminant_classifier(W5, x_train)
    display_CM_Error(y_pred_train, y_train, iris_legends)
    #"""

    print("\n-------------------------------------------------------------\n")

        
        