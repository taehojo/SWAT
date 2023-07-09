import argparse
import pandas as pd
from scripts.utils import load_data, impute_data, preprocess_data
from scripts.classifier import train_and_test_windows, train_and_test_selected_features
from scripts.visualization import plot_accuracies, plot_feature_importances
from scripts.api import get_snp_details
import os
import datetime
import numpy as np

def main(file, window_size, imputation_method, num_results, num_parallel_jobs):
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

    # Train and test for each window
    accuracies, window_indices = train_and_test_windows(features, labels, window_size, num_parallel_jobs)
    
    # Visualize results
    plot_accuracies(window_indices, accuracies, f'results/accuracies_{timestamp}.png')

    # Save accuracy results to CSV
    accuracy_df = pd.DataFrame(list(zip(window_indices, accuracies)), columns=['window_index', 'accuracy'])
    accuracy_df.to_csv(f'results/accuracy_results_{timestamp}.csv', index=False)

    # Top 'num_results' accuracy window indices
    window_accuracies = list(zip(accuracies, window_indices))
    top_accuracies = sorted(window_accuracies, key=lambda x: x[0], reverse=True)[:num_results]
    top_window_indices = [acc[1] for acc in top_accuracies]
    top_window_indices.sort()

    feature_indices = []
    for i in top_window_indices:
        feature_indices.extend(list(range(i, min(i+window_size, len(features.columns)))))
    selected_features = features.iloc[:, feature_indices]

    clf = train_and_test_selected_features(selected_features, labels)

    feature_importances = clf.feature_importances_
    feature_names = features.columns[feature_indices]

    top_indices = feature_importances.argsort()[-num_results:][::-1]
    top_features = [(feature_names[i], feature_importances[i]) for i in top_indices]

    # Get the top 'num_results' features
    enriched_top_features = get_snp_details(top_features)

    # Save enriched features to CSV
    pd.DataFrame(enriched_top_features).to_csv(f'results/top_{num_results}_features_{timestamp}.csv', index=False)

    # Plot feature importances
    plot_feature_importances(feature_importances, feature_names, f'results/feature_importances_{timestamp}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform SNP Analysis')
    parser.add_argument('input_file', type=str, help='Path to the input data file')
    parser.add_argument('--win', type=int, default=200, help='Window size for analysis')
    parser.add_argument('--imputation', type=str, default='simple', help='Imputation method. Choose from "simple", "1nn", "5nn", "10nn"')
    parser.add_argument('--num_results', type=int, default=20, help='Number of top results to output.')
    parser.add_argument('--num_jobs', type=int, default=-1, help='Number of jobs to run in parallel. -1 means using all processors.')

    args = parser.parse_args()
    main(args.input_file, args.win, args.imputation, args.num_results, args.num_jobs)
