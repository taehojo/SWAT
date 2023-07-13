import argparse
import pandas as pd
from scripts.utils import load_data, impute_data, preprocess_data
from scripts.classifier import train_and_test_windows, train_and_test_selected_features
from scripts.visualization import plot_accuracies, plot_feature_importances
from scripts.api import get_snp_details
import os
import datetime
import numpy as np
import sys

def main(file, window_size, imputation_method, num_results, num_parallel_jobs, classifier, no_plots, fast_run, api, WGS_select, WGS_merge, result_name=None):
    
    if not os.path.exists('results'):
        os.makedirs('results')

    timestamp = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')

    if result_name:
        timestamp = result_name

    data = load_data(file)
    data_imputed = impute_data(data, imputation_method)
    
    features, labels = preprocess_data(data_imputed, data)
    labels = labels.astype('int')

    if not WGS_merge:
        accuracies, window_indices = train_and_test_windows(features, labels, window_size, num_parallel_jobs, classifier)
        
        if not no_plots:
            plot_accuracies(window_indices, accuracies, f'results/accuracies_{timestamp}.png')

        accuracy_df = pd.DataFrame(list(zip(window_indices, accuracies)))
        accuracy_df.to_csv(f'results/accuracy_results_{timestamp}.csv', index=False, header=False)

        window_accuracies = list(zip(accuracies, window_indices))
        top_accuracies = sorted(window_accuracies, key=lambda x: x[0], reverse=True)[:num_results]
        
        top_window_indices = [acc[1] for acc in top_accuracies]
        top_window_indices.sort()

        if WGS_select:
            return

    else:
        try:
            top_window_indices_df = pd.read_csv(WGS_merge, header=None)
            top_window_indices = top_window_indices_df.iloc[:, 0].tolist()
        except FileNotFoundError:
            print(f"No saved top accuracies found at {WGS_merge}. Please run the script with '--WGS_select' option first.")
            sys.exit(1)

    feature_indices = []
    for i in top_window_indices:
        feature_indices.extend(list(range(i, min(i+window_size, len(features.columns)))))
    selected_features = features.iloc[:, feature_indices]

    clf_rf, clf_dl, feature_importances = train_and_test_selected_features(selected_features, labels, fast_run)
    
    feature_names = selected_features.columns

    top_indices = feature_importances.argsort()[-num_results:][::-1]
    top_features = [(feature_names[i], feature_importances[i]) for i in top_indices]

    if api:
        enriched_top_features = get_snp_details(top_features)
        top_features_df = pd.DataFrame(enriched_top_features)
    else:
        top_features_df = pd.DataFrame(top_features)
    
    top_features_df.to_csv(f'results/feature_importances_{timestamp}.csv', index=False, header=False)

    if not no_plots:
        plot_feature_importances(feature_importances, feature_names, f'results/feature_importances_{timestamp}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform SNP Analysis')
    parser.add_argument('input_file', type=str, help='Path to the input data file')
    parser.add_argument('--win', type=int, default=200, help='Window size for analysis')
    parser.add_argument('--imputation', type=str, default='5nn', help='Imputation method. Choose from "simple", "1nn", "5nn", "10nn"')
    parser.add_argument('--num_results', type=int, default=20, help='Number of top results to output.')
    parser.add_argument('--num_jobs', type=int, default=-1, help='Number of jobs to run in parallel. -1 means using all processors.')
    parser.add_argument('--classifier', type=str, default='rf', help='Classifier to use. Choose from "rf" for RandomForest and "dl" for Deep Learning.')
    parser.add_argument('--no_plots', action='store_true', help='If given, no plot images will be created.')
    parser.add_argument('--fast_run', action='store_true', help='If given, the script will run only RandomForest classifier and no plot images will be created.')
    parser.add_argument('--name', type=str, help='Name to use for the output files instead of the timestamp.')
    parser.add_argument('--no_api', dest='api', action='store_false', help='If given, the script will not make API calls to get SNP details.')
    parser.add_argument('--WGS_select', action='store_true', help='If given, the script will save top accuracies to a CSV file for later use.')
    parser.add_argument('--WGS_merge', type=str, help='If given, the script will load top accuracies from a specified CSV file and continue the analysis from that point.')

    args = parser.parse_args()
    
    if args.fast_run:
        args.no_plots = True
        args.classifier = 'rf'
    
    main(args.input_file, args.win, args.imputation, args.num_results, args.num_jobs, args.classifier, args.no_plots, args.fast_run, args.api, args.WGS_select, args.WGS_merge, args.name)
