from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
from joblib import Parallel, delayed
import numpy as np

def train_and_test_windows(features, labels, window_size, num_parallel_jobs, classifier):  
    def train_test_window(start):
        if start + window_size > len(features.columns):
            window_features = features.iloc[:, start:]
        else:
            window_features = features.iloc[:, start:start+window_size]
        
        train_features, test_features, train_labels, test_labels = train_test_split(window_features, labels, test_size=0.2, random_state=77)
        
        if classifier == 'dl':
            import tensorflow as tf

            n_samples = (len(train_features) // window_features.shape[1]) * window_features.shape[1]

            train_features = train_features.iloc[:n_samples]
            train_labels = train_labels.iloc[:n_samples]

            n_samples_test = (len(test_features) // window_features.shape[1]) * window_features.shape[1]
            test_features = test_features.iloc[:n_samples_test]
            test_labels = test_labels.iloc[:n_samples_test]

            clf = tf.keras.models.Sequential()
            clf.add(tf.keras.layers.Conv1D(32, 2, activation="relu", input_shape=(window_features.shape[1], 1)))
            clf.add(tf.keras.layers.Flatten())
            clf.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            clf.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

            train_features = np.array(train_features).reshape(-1, window_features.shape[1], 1)
            test_features = np.array(test_features).reshape(-1, window_features.shape[1], 1)
            
            clf.fit(train_features, train_labels, epochs=10, batch_size=32, verbose=0)
        elif classifier == 'rf':
            clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
            clf.fit(train_features, train_labels)

        pred_labels = (clf.predict(test_features) > 0.5).astype("int32") if classifier == 'dl' else clf.predict(test_features)
        accuracy = accuracy_score(test_labels, pred_labels)
        return start+1, accuracy

    results = Parallel(n_jobs=num_parallel_jobs)(delayed(train_test_window)(i) for i in range(0, len(features.columns), window_size))
    window_indices, accuracies = zip(*results)

    return accuracies, window_indices

def train_and_test_selected_features(features, labels):

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=77)

    scaler = MinMaxScaler()

    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=77)
    clf_rf.fit(train_features, train_labels)
    rf_predictions = clf_rf.predict(test_features)
    rf_accuracy = accuracy_score(test_labels, rf_predictions)
    feature_importances_rf = clf_rf.feature_importances_

    clf_dl = TabNetClassifier()
    clf_dl.fit(
        train_features, train_labels,
        eval_set=[(train_features, train_labels), (test_features, test_labels)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=500, patience=5,
        batch_size=128, virtual_batch_size=64,
        num_workers=0,
        weights=1,
        drop_last=False
    )
    dl_predictions = clf_dl.predict(test_features)
    dl_accuracy = accuracy_score(test_labels, dl_predictions)
    feature_importances_dl = clf_dl.feature_importances_

    if rf_accuracy > dl_accuracy:
        feature_importances = feature_importances_rf
    else:
        feature_importances = feature_importances_dl

    return clf_rf, clf_dl, feature_importances
