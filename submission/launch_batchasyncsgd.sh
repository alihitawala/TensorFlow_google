#!/bin/bash
source tfdefs_async.sh # this file is different from the one given originally - check grader folder for differences
start_cluster startserver_async.py # this file is different from the one given originally - check grader folder for differences

# start multiple clients
nohup python batchasyncsgd.py --task_index=0 > batchasynclog-0.out 2>&1&
sleep 10 # wait for variable to be initialized
nohup python batchasyncsgd.py --task_index=1 > batchasynclog-1.out 2>&1&
nohup python batchasyncsgd.py --task_index=2 > batchasynclog-2.out 2>&1&
nohup python batchasyncsgd.py --task_index=3 > batchasynclog-3.out 2>&1&
nohup python batchasyncsgd.py --task_index=4 > batchasynclog-4.out 2>&1&
nohup python batchasyncsgd.py --task_index=5 > batchasynclog-5.out 2>&1&
nohup python batchasyncsgd.py --task_index=6 > batchasynclog-6.out 2>&1&
nohup python batchasyncsgd.py --task_index=7 > batchasynclog-7.out 2>&1&
nohup python batchasyncsgd.py --task_index=8 > batchasynclog-8.out 2>&1&
nohup python batchasyncsgd.py --task_index=9 > batchasynclog-9.out 2>&1&
nohup python batchasyncsgd.py --task_index=10 > batchasynclog-10.out 2>&1&
nohup python batchasyncsgd.py --task_index=11 > batchasynclog-11.out 2>&1&
nohup python batchasyncsgd.py --task_index=12 > batchasynclog-12.out 2>&1&
nohup python batchasyncsgd.py --task_index=13 > batchasynclog-13.out 2>&1&
nohup python batchasyncsgd.py --task_index=14 > batchasynclog-14.out 2>&1&
nohup python batchasyncsgd.py --task_index=15 > batchasynclog-15.out 2>&1&
nohup python batchasyncsgd.py --task_index=16 > batchasynclog-16.out 2>&1&
nohup python batchasyncsgd.py --task_index=17 > batchasynclog-17.out 2>&1&
nohup python batchasyncsgd.py --task_index=18 > batchasynclog-18.out 2>&1&
nohup python batchasyncsgd.py --task_index=19 > batchasynclog-19.out 2>&1&
