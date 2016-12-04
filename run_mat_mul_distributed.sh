#!/bin/bash

# tfdefs.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source tfdefs.sh

export TF_LOG_DIR="/home/ubuntu/tf/logs"

# startserver.py has the specifications for the cluster.
start_cluster startserver.py


echo "Executing the distributed Matrix multiplication with d = " $1
# testdistributed.py is a client that can run jobs on the cluster.
# please read testdistributed.py to understand the steps defining a Graph and
# launch a session to run the Graph
st=$(date +%s)
python bigmatrixmultiplication.py $1
ed=$(date +%s)
time_taken=`expr $ed - $st`
echo $time_taken
# defined in tfdefs.sh to terminate the cluster
terminate_cluster

# Under the "GRAPHS" tab, use the options on the left to navigate to the "Run" you are interested in.
tensorboard --logdir=$TF_LOG_DIR


