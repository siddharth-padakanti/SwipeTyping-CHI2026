#!/bin/bash

for i in $(seq 3 9);
do
    echo accelerate launch findtuning_fold_update.py 2 $i
    accelerate launch findtuning_fold_update.py 2 $i
done