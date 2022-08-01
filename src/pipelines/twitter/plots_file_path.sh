#!/bin/bash

file_names=("BERT_music.pkl"
            "BERT_not_music.pkl"
            "BERT_no_rts_music.pkl"
            "BERT_no_rts_not_music.pkl");         

for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 dist_over_time_subplots.py $my_file_name

done