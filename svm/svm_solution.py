import numpy as np
from sklearn.svm import SVC


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    # Создание модели SVC
    model = SVC(C=1.5, class_weight={1: 5.5, 0: 8}, gamma='scale', kernel='rbf')
    model.fit(train_features, train_target)

    # Предсказание на тестовых данных
    predictions = model.predict(test_features)
    return predictions
