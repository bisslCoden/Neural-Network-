from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings
# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """

    # TODO: Create a PCA object and fit it using X_train
    #       Transform X_train using the PCA object.
    #       Print the explained variance ratio of the PCA object.
    #       Return both the transformed data and the PCA object.
    pca = PCA(n_components= n_components, random_state=42)
    pca.fit(X=X_train)
    x_reduces = pca.transform(X_train)
    print(f"Explained Variance by PCA: {np.sum(pca.explained_variance_ratio_) * 100}")
    #print(pca.components_.shape)
    return x_reduces, pca


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    print(f"{X_train.shape} we have here and {y_train.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Train MLPClassifier with different number of neurons in one hidden layer.
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration.
    #       Return the MLPClassifier that you consider to be the best.
    num_hidden = [2,10,100,200,500]
    best_classi = None
    for n_hidden in num_hidden:
        classifier = MLPClassifier(hidden_layer_sizes=(n_hidden,), max_iter=500, solver="adam",random_state=1)
        classifier.fit(X_train, y_train)
        acc_t = classifier.score(X_train, y_train)
        acc_v = classifier.score(X_val, y_val)
        
        #in my opinion the class with 200 neurons is the best because its still time efficient and has almost no 
        #difference to the one with 500
        if n_hidden == 200:
            best_classi = classifier
        print(f"{n_hidden} hidden neurons acc train: {acc_t:.2f}, acc val: {acc_v:.2f}, final loss {classifier.loss_:.2f}")

    return best_classi


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.
    
    # classifier_a = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, solver="adam",random_state=1, alpha=0.1)
    # classifier_a.fit(X_train, y_train)
    # return classifier_a
    
    num_hidden = [2,10,100,200,500]
    best_classi = None
    best_acc = 0.0

    #Set to false to save time and just take the best parameter combination (a) which I found when trying this code out and 
    #skip to testing just layer sizes (see below)
    exec_1_1_4 = True

    if exec_1_1_4:
        for n_hidden in num_hidden:
            classifier_a = MLPClassifier(hidden_layer_sizes=(n_hidden,), max_iter=500, solver="adam",random_state=1, alpha=0.1)
            classifier_a.fit(X_train, y_train)
            acc_t = classifier_a.score(X_train, y_train)
            acc_v = classifier_a.score(X_val, y_val)
            print(f"METHOD a) {n_hidden} hidden neurons acc train: {acc_t:.2f}, acc val: {acc_v:.2f}")
            if acc_v > best_acc:
                best_acc = acc_v
                best_classi = classifier_a
                print(f"SET net best classi to {n_hidden} with Method a)")

            classifier_b = MLPClassifier(hidden_layer_sizes=(n_hidden,), max_iter=500, solver="adam",random_state=1, early_stopping=True)
            classifier_b.fit(X_train, y_train)
            acc_t = classifier_b.score(X_train, y_train)
            acc_v = classifier_b.score(X_val, y_val)
            print(f"METHOD b) {n_hidden} hidden neurons acc train: {acc_t:.2f}, acc val: {acc_v:.2f}")
            if acc_v > best_acc:
                best_acc = acc_v
                best_classi = classifier_a
                print(f"SET net best classi to {n_hidden} with Method b)")
            
            classifier_c = MLPClassifier(hidden_layer_sizes=(n_hidden,), max_iter=500, solver="adam",random_state=1, alpha=0.1, early_stopping=True)
            classifier_c.fit(X_train, y_train)
            acc_t = classifier_c.score(X_train, y_train)
            acc_v = classifier_c.score(X_val, y_val)
            print(f"METHOD c) {n_hidden} hidden neurons acc train: {acc_t:.2f}, acc val: {acc_v:.2f}")
            if acc_v > best_acc:
                best_acc = acc_v
                best_classi = classifier_a
                print(f"SET net best classi to {n_hidden} with Method c)")
        
    print("best net gives: ",best_classi.score(X_val, y_val))
    print(best_classi.get_params())

    num_hidden = [2,10,100,200,500]
    best_classi = None
    for n_hidden in num_hidden:
        classifier = MLPClassifier(hidden_layer_sizes=(n_hidden,), max_iter=500, solver="adam",random_state=1, alpha=0.1)
        classifier.fit(X_train, y_train)
        acc_t = classifier.score(X_train, y_train)
        acc_v = classifier.score(X_val, y_val)
        if n_hidden == 200:
            best_classi = classifier
        print(f"Regularized Version: {n_hidden} hidden neurons acc train: {acc_t:.2f}, acc val: {acc_v:.2f}, final loss {classifier.loss_:.2f}")

    return best_classi


def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    # TODO: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes.
    plt.plot(nn.loss_curve_)
    plt.title("Loss Curve of best classifier (hidden_n = 200, alpha = 0.1)")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.show()


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use `nn` to compute predictions on `X_test`.
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data.
    #       Use `classification_report` to print the classification report.
    y_predict = nn.predict(X_test)
    acc_t = nn.score(X_test, y_test)
    print(f"final test accuracy: {acc_t}:.4f")
    print(classification_report(y_test, y_predict))
    confuse = confusion_matrix(y_test, y_predict)
    displ = ConfusionMatrixDisplay(confusion_matrix=confuse, display_labels=nn.classes_)
    displ.plot()
    plt.show()

def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.
    params = {
        'hidden_layer_sizes' : [(100,),(200,)],
        'alpha': [ 0.0, 0.1, 1.0 ],
        'solver' : [ 'lbfgs', 'adam' ]
    }
    network = MLPClassifier(max_iter=100, random_state=42)
    gs = GridSearchCV(network, params, cv=5, verbose=4)
    gs.fit(X_train, y_train)
    
    print(gs.best_score_, gs.best_params_)

    return gs.best_estimator_