import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

def load_data(file):
    data = pd.read_csv(file, sep=" ", header=0)
    data = data.iloc[:, 5:]
    return data.dropna(axis=1, how='all')

def impute_data(data, imputation_method):
    if imputation_method == 'simple':
        imputer = SimpleImputer(strategy='mean')
    elif imputation_method == '1nn':
        imputer = KNNImputer(n_neighbors=1)
    elif imputation_method == '5nn':
        imputer = KNNImputer(n_neighbors=5)
    elif imputation_method == '10nn':
        imputer = KNNImputer(n_neighbors=10)
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

def preprocess_data(data, window_size):
    features = data.iloc[:, 1:]
    labels = data.iloc[:, 0]
    labels = labels.apply(lambda x: 'rs_ith' if x == '.' else x)
    return features, labels

