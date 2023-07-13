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

## Requirements
- Python 3.8 or higher

## Usage
The tool can be run from the command line with the following syntax:

```bash
python main.py [input_file] --win [window_size] --imputation [imputation_method] --num_results [num_top_results] --num_jobs [num_parallel_jobs] --classifier [classifier] --name [output_file_name] --fast_run --no_plots --no_api --WGS_select --WGS_merge [merged_file_path]

```

where:

- `[input_file]` is the path to the input data file. This parameter is required.
- `[window_size]` is the window size for analysis. The default size is 200.
- `[imputation_method]` is the method employed to handle missing data. The options include "simple", "1nn", "5nn", or "10nn". "simple" stands for mean imputation, and "1nn", "5nn", "10nn" denote k-Nearest Neighbors method with k being 1, 5, and 10 respectively. The default method is "5nn".
- `[num_top_results]` determines the number of top results to output. The default is 20.
- `[num_parallel_jobs]` specifies the number of jobs to run in parallel. -1 means utilizing all processors. The default is to use all processors.
- `[classifier]` indicates the classifier to use. Choose "rf" for RandomForest and "dl" for Deep Learning. The default is "rf".
- `[output_file_name]` allows to choose a name for the output files instead of the timestamp.
- `--fast_run` is an option to execute the script only with the RandomForest classifier without creating plot images.
- `--no_plots` is an option to prevent the creation of plot images.
- `--no_api` is an option to prevent the script from making API calls to get SNP details.
- `--WGS_select` is an option to have the script save top accuracies to a CSV file for later use.
- `[merged_file_path]` is the path to a CSV file from which the script can load top accuracies and continue the analysis.


Execution example:
```bash
python main.py sample/APOE_LD_Block.csv
```

This command initiates the SNP analysis and stores the results in the 'results' directory. The outcomes include CSV files with the top N features and accuracy results, and if not suppressed, PNG files depicting accuracies and feature importances. Here N refers to the number of top results specified.

To execute the script for WGS files, you can use the provided bash script as follows:
```bash
run_swat.sh [input_file] [chunk_size]
```
For example:
```bash
./run_swat.sh sample/APOE_LD_Block.csv 1000
```

This will handle large WGS files by breaking them into smaller chunks, running the SNP analysis on each chunk, and then merging the results.

:bookmark: **SWAT citation:**

> (in preparation)


:bookmark: **Example of SWAT application:**

> Jo, Taeho, et al. "Deep learning-based identification of genetic variants: application to Alzheimer’s disease classification." Briefings in Bioinformatics 23.2 (2022)


