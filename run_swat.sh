#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: ./run_swat.sh [input_file] [chunk_size]"
    exit 1
fi

INPUT_FILE=$1
CHUNK_SIZE=$2
RESULTS_DIR="results"
FIRST_COL="first_column_tmp.csv"

mkdir -p $RESULTS_DIR

TOTAL_COLS=$(awk -F' ' '{print NF; exit}' $INPUT_FILE)

cut -d' ' -f1-6 $INPUT_FILE > $RESULTS_DIR/$FIRST_COL

for ((i=7; i<=$TOTAL_COLS; i=i+$CHUNK_SIZE))
do
    CHUNK_NAME="tmp-$i.csv"
    OUTPUT_NAME="$i"

    cut -d' ' -f$i-$((i+$CHUNK_SIZE-1)) $INPUT_FILE > $RESULTS_DIR/tmp.csv

    paste -d' ' $RESULTS_DIR/$FIRST_COL $RESULTS_DIR/tmp.csv > $RESULTS_DIR/$CHUNK_NAME
 
    python main.py $RESULTS_DIR/$CHUNK_NAME --fast_run --no_api --WGS_select --name tmp_WGS_$OUTPUT_NAME

    awk -v offset=$OUTPUT_NAME -F"," 'BEGIN {OFS = ","} {$1+=offset; print $0}' $RESULTS_DIR/accuracy_results_tmp_WGS_$OUTPUT_NAME.csv > $RESULTS_DIR/accuracy_results_tmp_WGS_edit_$OUTPUT_NAME.csv

done

cat $RESULTS_DIR/accuracy_results_tmp_WGS_edit_*.csv > $RESULTS_DIR/accuracy_results_tmp_WGS.csv

python main.py $INPUT_FILE --fast_run --no_api --WGS_merge $RESULTS_DIR/accuracy_results_tmp_WGS.csv --name WGS_FINAL --num_results 1000

rm $RESULTS_DIR/*tmp*.csv
