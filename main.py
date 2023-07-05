import argparse
import pandas as pd
from utils import load_data, impute_data, preprocess_data
from classifier import train_and_test
from visualization import plot_accuracies, plot_feature_importances
from api import get_snp_details

def main(file, window_size, imputation_method):
    data = load_data(file)
    data_imputed = impute_data(data, imputation_method)
    features, labels = preprocess_data(data_imputed, window_size)
    clf, accuracy = train_and_test(features, labels)
    # Remaining code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform SNP Analysis')
    parser.add_argument('--file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--window_size', type=int, required=True, help='Window size for analysis')
    parser.add_argument('--imputation_method', type=str, required=True, help='Imputation method. Choose from "simple", "1nn", "5nn", "10nn"')

    args = parser.parse_args()

    main(args.file, args.window_size, args.imputation_method)

