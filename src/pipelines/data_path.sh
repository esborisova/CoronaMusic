#!/bin/bash

file_names=("music.pkl"
            "not_music.pkl");         

for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 BERT.py $my_file_name

done