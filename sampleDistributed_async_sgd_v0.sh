#!/bin/bash
source tfdefs.sh
start_cluster startserver.py
# start multiple clients
nohup python asynchronoussgd_distributed.py --task_index=0 > asynclog-0.out 2>&1&
sleep 2 # wait for variable to be initialized
nohup python asynchronoussgd_distributed.py --task_index=1 > asynclog-1.out 2>&1&
nohup python asynchronoussgd_distributed.py --task_index=2 > asynclog-2.out 2>&1&
nohup python asynchronoussgd_distributed.py --task_index=3 > asynclog-3.out 2>&1&
nohup python asynchronoussgd_distributed.py --task_index=4 > asynclog-4.out 2>&1&
# defined in tfdefs.sh to terminate the cluster
terminate_cluster
