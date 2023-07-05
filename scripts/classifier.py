from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_test(features, labels, test_size=0.2, random_state=42, n_jobs=-1):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=random_state, n_jobs=n_jobs)
    clf.fit(train_features, train_labels)
    pred_labels = clf.predict(test_features)
    return clf, accuracy_score(test_labels, pred_labels)

