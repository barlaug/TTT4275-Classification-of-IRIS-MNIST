from distutils.log import error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, zero_one_loss
import iris


def display_ER_MSE_iterations(x, targets, train_set_size, num_classes):
    """Displays plots for Error rate vs iterations and MSE vs iterations, 
        respectively, for different alphas
    Params:
        x: np.ndarray
            Complete iris dataset
        targets: np.ndarray
            Complete iris dataset
        x: np.ndarray
            Complete iris dataset
        num_classes: int
            Number of classes
    """
    x_train, x_test, t_train, t_test, y_train, y_test = iris.extract_sets(x, targets, train_set_size, num_classes)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    #Values for which we want to test
    alphas = [0.0005, 0.001, 0.005, 0.05]
    iterations = 3000
    for alpha in alphas:
        print("Alpha: " + str(alpha))
        error_rate_set = []
        alpha_label = "alpha="+ str(alpha)
        W1, W_set, iter_set, MSE_set = iris.train_lin_model(x_train[:,:], t_train, alpha, iterations, num_classes)
        for W in W_set:
            y_pred = iris.discriminant_classifier(W, x_test)
            error_rate = iris.get_error_rate(y_test, y_pred)
            error_rate_set.append(error_rate)
        error_rate_vec = np.array(error_rate_set)
        iteration_vec = np.array(iter_set)
        MSE_vec = np.array(MSE_set)
        #Plotting:
        ax1.plot(iteration_vec, error_rate_vec, label=alpha_label)
        ax1.legend(fontsize=12)
        ax2.plot(iteration_vec, MSE_vec[:,0,0], label=alpha_label)
        ax2.legend(fontsize=12)
    ax1.set_ylabel("Error rate", fontsize=14)
    ax2.set_ylabel("MSE", fontsize=14)
    ax2.set_xlabel("Iterations", fontsize=14)
    ax1.set_title("Error rate vs Iterations for different alpha", fontsize=16)
    ax2.set_title("MSE vs Iterations for different alpha", fontsize=16)
    print('Plotting ER and MSE vs. iterations...\n')
    #plt.savefig('error_rate_MSE_vs_iterations2.eps', format='eps')
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
    print('Displaying confusion matrix...\n')
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
        plt.hist(x[N:2*N,i], alpha=0.5, label=labels[1])
        plt.hist(x[2*N:len(x),i], alpha=0.5, label=labels[2])
        plt.legend()
        plt.xlabel(features[i])
        plt.ylabel("Count")
        plt.xlabel("%s [cm]" % features[i])
    #plt.savefig('histogram.eps', format='eps')
    print("Plotting histogram...\n")
    plt.show()
