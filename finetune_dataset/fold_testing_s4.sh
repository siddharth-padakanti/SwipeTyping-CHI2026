#!/bin/bash

for i in $(seq 4 9);
do
    echo accelerate launch findtuning_fold.py 4 $i
    accelerate launch findtuning_fold.py 4 $i
done