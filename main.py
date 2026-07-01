import numpy as np
from sklearn.model_selection import train_test_split
from nn_classification_from_scratch import train_nn_own
from nn_classification_sklearn import reduce_dimension, train_nn, train_nn_with_regularization, \
    perform_grid_search, show_confusion_matrix_and_classification_report, plot_training_loss_curve


def run_sklearn_baseline():
    # Load dataset
    features = np.load('data/sign_language_images.npy')
    targets = np.load('data/sign_language_targets.npy')
    features = features.reshape((features.shape[0], -1))
    print(features.shape, targets.shape)

    X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=0.2, random_state=42)

    print(features.shape)

    # PCA-based dimensionality reduction
    print("----- PCA Reduction -----")
    n_components = 310
    X_train_pca, pca = reduce_dimension(X_train, n_components)
    print(X_train_pca.shape)

    # Baseline neural network training
    print("----- Baseline Model Training -----")
    best_nn = train_nn(X_train_pca, y_train)

    # Regularized model training
    print("----- Regularized Training -----")
    best_reg_nn = train_nn_with_regularization(X_train_pca, y_train)

    best_model = best_reg_nn
    plot_training_loss_curve(best_model)

    # Hyperparameter search
    print("----- Hyperparameter Optimization -----")
    best_gs_nn = perform_grid_search(X_train_pca, y_train)

    final_model = best_gs_nn

    X_test_pca = pca.transform(X_test)
    show_confusion_matrix_and_classification_report(final_model, X_test_pca, y_test)


def run_custom_multiclass_model():
    features = np.load('data/sign_language_images.npy')
    targets = np.load('data/sign_language_targets.npy')
    features = features.reshape((features.shape[0], -1))

    X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=0.2, random_state=42)
    X_train_pca, pca = reduce_dimension(X_train, n_components=16)

    nn = train_nn_own(X_train_pca, y_train)

    X_test_pca = pca.transform(X_test)
    test_acc = nn.score(X_test_pca, y_test)
    print(f'Test accuracy: {test_acc:.4f}.')


def run_custom_binary_model():
    features = np.load('data/sign_language_images.npy')
    targets = np.load('data/sign_language_targets.npy')
    features = features.reshape((features.shape[0], -1))

    # Create binary classification dataset
    idxs = np.where(targets < 2)[0]
    features = features[idxs]
    targets = targets[idxs]

    X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=0.2, random_state=42)
    X_train_pca, pca = reduce_dimension(X_train, n_components=16)

    nn = train_nn_own(X_train_pca, y_train)

    X_test_pca = pca.transform(X_test)
    test_acc = nn.score(X_test_pca, y_test)
    print(f'Test accuracy: {test_acc:.4f}.')


def main():
    run_sklearn_baseline()
    run_custom_multiclass_model()
    run_custom_binary_model()


if __name__ == '__main__':
    main()