#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <path_to_test_graph> <path_to_model> <output_file_path>"
  exit 1
fi

dataset_folder=$1
model_path=$2
submission_path=$3


python3 src/testing_2.py $dataset_folder $model_path $submission_path