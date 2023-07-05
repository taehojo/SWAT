import matplotlib.pyplot as plt

def plot_accuracies(window_indices, accuracies):
    plt.figure(figsize=(5, 5))
    plt.plot(window_indices, accuracies)
    plt.xticks(rotation=0)
    plt.title('Accuracy per window')
    plt.xlabel('Window position')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('accuracy_per_window.png')

def plot_feature_importances(feature_importances, feature_names):
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(feature_importances)), feature_importances, width=1.5, tick_label=feature_names)
    plt.xticks(np.arange(0, len(feature_importances), 1000), np.arange(0, len(feature_importances), 1000))
    plt.title('Feature importances')
    plt.xlabel('SNP position')
    plt.ylabel('Importance')
    plt.tight_layout()

    top_5_indices = feature_importances.argsort()[-5:][::-1]
    for i in top_5_indices:
        plt.text(i, feature_importances[i], feature_names[i], ha='center', va='bottom')
    
    plt.savefig('feature_importances.png')

