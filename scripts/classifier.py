from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import numpy as np

def train_and_test_windows(features, labels, window_size):
    def train_test_window(start):
        if start + window_size > len(features.columns):
            window_features = features.iloc[:, start:]
        else:
            window_features = features.iloc[:, start:start+window_size]
        train_features, test_features, train_labels, test_labels = train_test_split(window_features, labels, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
        clf.fit(train_features, train_labels)
        pred_labels = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, pred_labels)
        return start+1, accuracy

    results = Parallel(n_jobs=-1)(delayed(train_test_window)(i) for i in range(0, len(features.columns), window_size))
    window_indices, accuracies = zip(*results)

    return accuracies, window_indices

def train_and_test_selected_features(features, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train_features, train_labels)

    return clf
