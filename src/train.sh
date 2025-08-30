#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <path_to_train_graph> <output_model_file_path>"
  exit 1
fi

dataset_folder=$1
model_path=$2

python3 src/training_2.py $dataset_folder $model_path