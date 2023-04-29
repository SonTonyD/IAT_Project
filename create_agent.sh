#!/bin/bash

# read the parameters from the text file
while read -r line; do
    # split the line into space-separated values
    read -ra params <<< "$line"
    # run the python script with the parameters
    python3 run_game.py "${params[@]}"
done < hyperparameter_test_set.txt
