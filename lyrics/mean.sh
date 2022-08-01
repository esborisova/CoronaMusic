#!/bin/bash

file_names=("BERT_sent_scores"
            "BERT_emot_scores");         

for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 mean_sent.py $my_file_name

done