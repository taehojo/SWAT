import argparse
import pandas as pd
from scripts.utils import load_data, impute_data, preprocess_data
from scripts.classifier import train_and_test
from scripts.visualization import plot_accuracies, plot_feature_importances
from scripts.api import get_snp_details
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracies(window_indices, accuracies, filename):
    plt.plot(window_indices, accuracies, marker='o')
    plt.xlabel('Window start index')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over window indices')
    plt.grid()
    plt.savefig(filename)
    plt.close()

def plot_feature_importances(feature_importances, feature_names, filename):
    plt.bar(range(len(feature_importances)), feature_importances, width=1.5, tick_label=feature_names)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature importances')
    plt.xticks(np.arange(0, len(feature_importances), 1000), np.arange(0, len(feature_importances), 1000))
    plt.savefig(filename)
    plt.close()

def main(file, window_size, imputation_method):
    # Create a results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Load and impute data
    data = load_data(file)
    data_imputed = impute_data(data, imputation_method)

    # Preprocess data
    features, labels = preprocess_data(data_imputed, data)

    # Train and test
    clf, accuracies, window_indices = train_and_test(features, labels, window_size)
    
    # Visualize results
    plot_accuracies(window_indices, accuracies, f'results/accuracies_{timestamp}.png')

    # Save accuracy results to CSV
    accuracy_df = pd.DataFrame(list(zip(window_indices, accuracies)), columns=['window_index', 'accuracy'])
    accuracy_df.to_csv(f'results/accuracy_results_{timestamp}.csv', index=False)
      
    # Derive the feature indices (considering window size) and get the corresponding feature names
    feature_indices = [idx for window_start in window_indices for idx in range(window_start, min(window_start+window_size, len(features.columns)))]
    feature_names = features.columns.tolist()
    
    # Get the feature importances from the classifier
    feature_importances = clf.feature_importances_
    
    # Get SNP details
    top_20_indices = feature_importances.argsort()[-20:][::-1]
    top_20_features = [(feature_names[i], feature_importances[i]) for i in top_20_indices]
    enriched_top_20_features = get_snp_details(top_20_features)

    # Save enriched features to CSV
    pd.DataFrame(enriched_top_20_features).to_csv(f'results/top_20_features_{timestamp}.csv', index=False)

    # Plot feature importances
    plot_feature_importances(feature_importances, feature_names, f'results/feature_importances_{timestamp}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform SNP Analysis')
    parser.add_argument('--file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--window_size', type=int, required=True, help='Window size for analysis')
    parser.add_argument('--imputation_method', type=str, required=True, help='Imputation method. Choose from "simple", "1nn", "5nn", "10nn"')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs to run in parallel. -1 means using all processors.')

    args = parser.parse_args()

    main(args.file, args.window_size, args.imputation_method)
