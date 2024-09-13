#!/bin/bash

neural_encoding=$1
# epoch=30
batch_size=16
lr=0.0001
# device=$2
time_step=100
data_root="/home/yschoi/snn/bin_cf/bin_classification/MIT_BIH_ECG/562_bin_mitbih_train.csv"
test_data_root="/home/yschoi/snn/bin_cf/bin_classification/MIT_BIH_ECG/562_bin_mitbih_test.csv"
save_dir="train_test_trace"
date="240313_561data_syn_stepsize_epoch"

current_path=`pwd`


if [ ${neural_encoding} = "HSA" ]
then

    save_out_file="${current_path}/${save_dir}/${date}_${neural_encoding}.txt"
    python bin_main.py -ne $neural_encoding -n syn -e 15 -b $batch_size -lr $lr --print_epoch 1 -nd 0 -t $time_step -s True --step_size 3 --data_root $data_root --test_data_root $test_data_root > $save_out_file
    python bin_test.py -ne $neural_encoding -n syn -e 15 -b $batch_size -lr $lr --print_epoch 1 -nd 0 -t $time_step -s True --step_size 3 --data_root $data_root --test_data_root $test_data_root >> $save_out_file

fi

if [ ${neural_encoding} = "BSA" ]
then

    save_out_file="${current_path}/${save_dir}/${date}_${neural_encoding}.txt"
    python bin_main.py -ne $neural_encoding -n syn -e 15 -b $batch_size -lr $lr --print_epoch 1 -nd 2 -t $time_step -s True --step_size 3 --data_root $data_root --test_data_root $test_data_root > $save_out_file
    python bin_test.py -ne $neural_encoding -n syn -e 15 -b $batch_size -lr $lr --print_epoch 1 -nd 2 -t $time_step -s True --step_size 3 --data_root $data_root --test_data_root $test_data_root >> $save_out_file

fi

if [ ${neural_encoding} = "BURST" ]
then

    save_out_file="${current_path}/${save_dir}/${date}_${neural_encoding}_repeat.txt"
    python bin_main.py -ne $neural_encoding -n syn -e 15 -b $batch_size -lr $lr --print_epoch 1 -nd 1 -t $time_step -s True --step_size 3 --data_root $data_root --test_data_root $test_data_root > $save_out_file
    python bin_test.py -ne $neural_encoding -n syn -e 15 -b $batch_size -lr $lr --print_epoch 1 -nd 1 -t $time_step -s True --step_size 3 --data_root $data_root --test_data_root $test_data_root >> $save_out_file

fi

# if [ ${neural_encoding} = "TTFS1" ]
# then

#     save_out_file="${save_dir}/${date}_${neural_encoding}.txt"
#     python bin_main.py -ne $neural_encoding -n syn -e 10 -b $batch_size -lr $lr --print_epoch 1 -nd 3 -t $time_step --time_step $time_step -s True --data_root $data_root --test_data_root $test_data_root > $save_out_file
#     python bin_test.py -ne $neural_encoding -n syn -e 10 -b $batch_size -lr $lr --print_epoch 1 -nd 3 -t $time_step --time_step $time_step -s True --data_root $data_root --test_data_root $test_data_root >> $save_out_file

# fi

if [ ${neural_encoding} = "TTFS" ] 
then

    save_out_file="${current_path}/${save_dir}/${date}_${neural_encoding}.txt"
    python bin_main.py -ne $neural_encoding -n syn -e 10 -b $batch_size -lr $lr --print_epoch 1 -nd 3 -t $time_step -s True --step_size 2 --data_root $data_root --test_data_root $test_data_root > $save_out_file
    python bin_test.py -ne $neural_encoding -n syn -e 10 -b $batch_size -lr $lr --print_epoch 1 -nd 3 -t $time_step -s True --step_size 2 --data_root $data_root --test_data_root $test_data_root >> $save_out_file

fi 