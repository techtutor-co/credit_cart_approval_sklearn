import json
import click
import pickle
from joblib import dump, load

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def label_encode(df, column):
    le = preprocessing.LabelEncoder()
    values = list(df[column].values)
    le.fit(values)
    df[column] = le.transform(values)
    return df


def one_hot_encode(df, column):
    df = df.join(pd.get_dummies(df[column], prefix=column))
    return df.drop([column], axis=1)


def scale_normalize(df, columns):
    df[columns] = MinMaxScaler().fit_transform(df[columns])
    for column in columns:
        df[column] = df[column].apply(lambda x: np.log(x + 1))
    return df


def encode_dataset(df):
    df.drop(['ID'], axis=1, inplace=True)
    df['Age'] = df['Age'].apply(lambda value: int(value / 10) * 10)

    df = one_hot_encode(df, 'Securities Account')
    df = one_hot_encode(df, 'CD Account')
    df = one_hot_encode(df, 'Online')
    df = one_hot_encode(df, 'CreditCard')

    df = label_encode(df, 'ZIP Code')
    features = df.drop(['Personal Loan'], axis=1)
    labels = df['Personal Loan']

    features = scale_normalize(features, ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education',
                                          'Mortgage'])
    df = features
    df['Personal Loan'] = labels
    return df


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    return accuracy_score(y, predictions)


def train_model(dataset, config):
    from sklearn.svm import SVC

    X = dataset.drop(['Personal Loan'], axis=1)
    y = dataset['Personal Loan']

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=config['validation_size'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'])

    learner = SVC()
    learner.set_params(**config['model_hyperparameters'])
    learner.fit(X_train, y_train)

    test_score = evaluate_model(learner, X_test, y_test)
    validation_score = evaluate_model(learner, X_validation, y_validation)
    return learner, test_score, validation_score


@click.command()
@click.option("--config_path")
def build_model(config_path):
    config = None
    with open(config_path) as json_data_file:
        config = dict(json.load(json_data_file))
    dataset = pd.read_csv(config['dataset_path'])
    dataset = encode_dataset(dataset)

    model, test_score, validation_score = train_model(dataset, config)
    dump(model, config['model_path'])
    print(f'test_score: {test_score}')
    print(f'validation_score: {validation_score}')


if __name__ == "__main__":
    build_model()
