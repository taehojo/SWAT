import argparse
from scripts.utils import load_data, impute_data, preprocess_data
from scripts.classifier import train_and_test
from scripts.visualization import plot_accuracies, plot_feature_importances
from scripts.api import get_snp_details

def main(file, window_size, imputation_method):
    # Load and impute data
    data = load_data(file)
    data_imputed = impute_data(data, imputation_method)

    # Preprocess data
    features, labels = preprocess_data(data_imputed, data, window_size)

    # Train and test
    clf, accuracies, window_indices = train_and_test(features, labels, window_size)
    
    # Visualize results
    plot_accuracies(window_indices, accuracies)
    feature_importances, feature_indices, feature_names = clf.feature_importances_, features.columns[window_indices], features.columns

    # Get SNP details
    top_20_features = [(feature_names[i], feature_importances[i]) for i in feature_importances.argsort()[-20:][::-1]]
    enriched_top_20_features = get_snp_details(top_20_features)

    # Save enriched features to CSV
    pd.DataFrame(enriched_top_20_features).to_csv('top_20_features.csv', index=False)

    # Plot feature importances
    plot_feature_importances(feature_importances, feature_indices, feature_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform SNP Analysis')
    parser.add_argument('--file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--window_size', type=int, required=True, help='Window size for analysis')
    parser.add_argument('--imputation_method', type=str, required=True, help='Imputation method. Choose from "simple", "1nn", "5nn", "10nn"')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs to run in parallel. -1 means using all processors.')

    args = parser.parse_args()

    main(args.file, args.window_size, args.imputation_method)
