#!/bin/bash
# OGA-TS model runner

export PYTHONPATH="`pwd`/:$PYTHONPATH"
source activate ai # activate conda environment

model=sed_ogits
log_dir=logs/$(date "+%d_%b_%Y")
log_file=$log_dir/${model}_$(date "+%H_%M_%S").log
mkdir $log_dir

echo "Log to: $log_file"
python -u run_ogits.py -m dev -a place -p ./params/ogits.yaml > ${log_file}_place

python -u run_ogits.py -m dev -a manner -p ./params/ogits.yaml > ${log_file}_manner

python -u run_ogits.py -m dev -a fusion -p ./params/ogits.yaml > ${log_file}_fusion
