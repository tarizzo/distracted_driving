from model import samples_per_time_series, x_features, y_feature, create_time_series_data
from sequentia.classifiers import KNNClassifier

if __name__ == "__main__":
    x, y = create_time_series_data('data/', samples_per_time_series, x_features, y_feature)

    x = list(x)
    y = y.toXlist()

    clf = KNNClassifier(k=1, classes=list(set(y)))
    clf.fit(x, y)

    clf.predict(x)