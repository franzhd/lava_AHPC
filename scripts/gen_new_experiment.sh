#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <root_folder>"
    exit 1
fi

experiment_folder=$"nni_experiments"
experiment_name="$experiment_folder/$1"
root_folder=$(realpath "$experiment_name")
parent_folder=$(dirname "$root_folder")

# Create the root folder
mkdir -p "$root_folder"

# Create the subdirectories and files
mkdir -p "$root_folder/code"
touch "$root_folder/code/experiment1.py"
echo "
DATASET_NAME = 
TRAIN_FOLDER_NAME = 'Trained'
NORM_TYPE =
NUM_WORKERS = 8
NET_OUTPUT_DIM = 
NET_INPUT_DIM = 
NUM_EPOCHS = 200
" >> "$root_folder/code/config.py"

mkdir -p "$root_folder/config"
echo " 
experimentName: $1
trialConcurrency: 2
maxExperimentDuration: 72h
maxTrialNumber: 1000
trainingService:
  platform: local
  useActiveGpu: True
  maxTrialNumberPerGpu: 4
searchSpaceFile: ~/lava_AHPC/nni_experiments/$1/search_space/search_space1.json
useAnnotation: false
tuner: 
    name: Anneal
    classArgs:
      optimize_mode: maximize


trialCodeDirectory: ~/lava_AHPC/nni_experiments/$1/code/
trialCommand: python3 experiment1.py --trial_path $1/
experimentWorkingDirectory: ~/lava_AHPC/nni_experiments/$1/results/
trialGpuNumber: 1

" >> "$root_folder/config/config_1.yml"

mkdir -p "$root_folder/results"
touch $root_folder/results/.gitkeep

mkdir -p "$root_folder/search_space"
echo "
{
    \"treshold\": {\"_type\": \"quniform\", \"_value\": [0, 5, 0.2]},
    \"voltage_decay\": {\"_type\": \"quniform\", \"_value\": [0,1, 0.05]},
    \"batch_size\": {\"_type\": \"quniform\", \"_value\": [16,256, 16]},
    \"lr\": {\"_type\": \"quniform\", \"_value\": [0.00001,0.001,0.000005]}


}" >> "$root_folder/search_space/search_space1.json"

echo "Directory structure created under $root_folder."
echo "Parent folder: $parent_folder"