#!/bin/bash

for i in $(seq 1 9);
do
    accelerate launch findtuning_fold.py $i
done