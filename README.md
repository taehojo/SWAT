# SWAT (Sliding Window Association Test) 

## Introduction

SWAT (Sliding Window Association Test) is a tool for Whole Genome Sequencing (WGS) analysis using machine learning. It's a newly developed Python-based tool that aims to provide a robust and efficient way to analyze high-dimensional genomic data.

SWAT is designed to identify phenotype-related single nucleotide polymorphisms (SNPs), making it particularly useful for developing accurate disease classification models. The tool also includes a sophisticated imputer which is capable of automatically filling in missing data, hence improving the quality of the analysis.

You can also access the web implementation of this tool at [SWAT-web](https://www.swatweb.org).

## Installation

To install and run this tool, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/taehojo/SWAT.git
```

2. Navigate to the project directory:

```bash
cd SWAT
```

3. Install the required Python packages. It's recommended to do this in a virtual environment:

```bash
pip install -r requirements.txt
```

## Usage
The tool can be run from the command line with the following syntax:

```bash
python main.py --input_file [input_file] --analysis_window_size [analysis_window_size] --missing_value_imputation_method [missing_value_imputation_method] --num_top_results [num_top_results]
```

where:

- `[input_file]` is the path to the input data file.
- `[window_size]` is the window size for analysis.
- `[imputation_method]` is the imputation method used for dealing with missing data. You can choose from "simple", "1nn", "5nn", or "10nn". Here "simple" represents mean imputation and "1nn", "5nn", "10nn" represents the k-Nearest Neighbors method with k being 1, 5, and 10 respectively.
- `[num_top_results]` is the number of top results to output. The default is 20.

For example:
```bash
python main.py --input_file sample/SampleWithAPOE --analysis_window_size 200 --missing_value_imputation_method simple --num_top_results 30
```

This will start the SNP analysis and save the results in the results directory. The results will include CSV files with the top N features and accuracy results, and PNG files with plots of accuracies and feature importances. Here N is the number of top results specified.


:bookmark: **SWAT citation:**

> in preparation


:bookmark: **Example of SWAT application:**

> Jo, Taeho, et al. "Deep learning-based identification of genetic variants: application to Alzheimer’s disease classification." Briefings in Bioinformatics 23.2 (2022): bbac022.


