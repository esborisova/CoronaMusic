#!/bin/bash

file_names=("raw_data/1_all_corona_songs/"
            "raw_data/2a_corona_parodies-20220725T071845Z-001/2a_corona_parodies/"
            "raw_data/2b_matched_originals-20220725T072105Z-001/2b_matched_originals");         

for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 save_data.py $my_file_name

done