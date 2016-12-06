#!/bin/bash
source tfdefs_multiple.sh # this file is different from the one given originally - check grader folder for differences
start_cluster startserver_multiple.py # this file is different from the one given originally - check grader folder for differences

# start multiple clients
nohup python asyncsgd.py --task_index=0 > asynclog-0.out 2>&1&
sleep 10 # wait for variable to be initialized
nohup python asyncsgd.py --task_index=1 > asynclog-1.out 2>&1&
nohup python asyncsgd.py --task_index=2 > asynclog-2.out 2>&1&
nohup python asyncsgd.py --task_index=3 > asynclog-3.out 2>&1&
nohup python asyncsgd.py --task_index=4 > asynclog-4.out 2>&1&
nohup python asyncsgd.py --task_index=5 > asynclog-5.out 2>&1&
nohup python asyncsgd.py --task_index=6 > asynclog-6.out 2>&1&
nohup python asyncsgd.py --task_index=7 > asynclog-7.out 2>&1&
nohup python asyncsgd.py --task_index=8 > asynclog-8.out 2>&1&
nohup python asyncsgd.py --task_index=9 > asynclog-9.out 2>&1&
nohup python asyncsgd.py --task_index=10 > asynclog-10.out 2>&1&
nohup python asyncsgd.py --task_index=11 > asynclog-11.out 2>&1&
nohup python asyncsgd.py --task_index=12 > asynclog-12.out 2>&1&
nohup python asyncsgd.py --task_index=13 > asynclog-13.out 2>&1&
nohup python asyncsgd.py --task_index=14 > asynclog-14.out 2>&1&
nohup python asyncsgd.py --task_index=15 > asynclog-15.out 2>&1&
nohup python asyncsgd.py --task_index=16 > asynclog-16.out 2>&1&
nohup python asyncsgd.py --task_index=17 > asynclog-17.out 2>&1&
nohup python asyncsgd.py --task_index=18 > asynclog-18.out 2>&1&
nohup python asyncsgd.py --task_index=19 > asynclog-19.out 2>&1&
