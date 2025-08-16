#!/bin/bash

for i in $(seq 1 9);
do
    echo accelerate launch findtuning_fold.py 2 $i
    accelerate launch findtuning_fold.py 2 $i
done