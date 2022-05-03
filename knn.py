from model import samples_per_time_series, x_features, y_feature, create_time_series_data
from sequentia.classifiers import KNNClassifier
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = create_time_series_data('data/', samples_per_time_series, x_features, y_feature)

    x_train = list(x_train)
    y_train = y_train.tolist()
    x_test = list(x_test)
    y_test = y_test.tolist()

    clf = KNNClassifier(k=3, classes=list(set(y_train)))
    clf.fit(x_train, y_train)

    y_hat = clf.predict(x_train)
    train_accuracy = accuracy_score(y_train, (y_hat).astype(np.float32))
    y_hat = clf.predict(x_test)
    testing_accuracy = accuracy_score(y_test, (y_hat).astype(np.float32))

    print("Training Accuracy: {}, Testing Accuracy: {}".format(train_accuracy, testing_accuracy))