from repository.datasource_db import DataSourceDatabase

# Add custom imports
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train(db: DataSourceDatabase, path_to_dir: str):

    model = define()

    # Get data from datasource as dataframe
    #
    # For successive training use:
    #
    #   batch_size = 100
    #   records_count = db.get_records_count()
    #   for offset in range(0, records_count, batch_size):
    #       batch_df = db.get_batch(batch_size, offset)
    #
    #
    # For instant training use:
    #
    #   df = db.get_all_records()

    batch_size = 100
    classes = [0, 1, 2]
    records_count = db.get_records_count()
    for offset in range(0, records_count, batch_size):
        batch_df = db.get_batch(batch_size, offset)
        # drop id column
        batch_df = batch_df.drop("id", axis=1)

        batch_df = transform_data(batch_df)
        print(batch_df)
        train_features, test_features, train_labels, test_labels = split_data(batch_df)

        try:
            train_data = {"features": train_features, "labels": train_labels}
            train_features = train_data["features"]
            train_labels = train_data["labels"]
            print(train_features)
            print(train_labels)
            print(classes)
            model.partial_fit(train_features, train_labels, classes=classes)
        except AttributeError:
            raise ValueError("It is only allowed to use models with 'partial_fit' method.")

    # Calculate model training accuracy
    # TODO: Maybe create more appropriate way of transferring data to validate method?
    test_data = {"features": test_features, "labels": test_labels}
    accuracy = validate(model, test_data)

    # Implement saving result model into `path_to_dir`
    save_model(model, path_to_dir)

    # Return accuracy
    return accuracy


def save_model(model, path_to_dir):
    file_name = "model.pkl"
    joblib.dump(model, "{}/{}".format(path_to_dir, file_name))


def define():
    model = SGDClassifier(loss="log", penalty="l2", max_iter=5)
    return model


def transform_data(data):
    index_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    mapped = data.replace({"class": index_mapping})
    return mapped


def split_data(data):
    features = data.drop("class", axis=1).values
    labels = np.array(data["class"].values)

    return train_test_split(features, labels, test_size=0.1)


def validate(model, test_data):
    test_features = test_data["features"]
    test_labels = test_data["labels"]

    predicted_iris_labels = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predicted_iris_labels)

    return accuracy * 100



