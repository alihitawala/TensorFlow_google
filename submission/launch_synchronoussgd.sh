#!/bin/bash

# tfdefs.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source tfdefs_multiple.sh

# startserver.py has the specifications for the cluster.
start_cluster startserver_multiple.py

echo "Executing the distributed synchronous sgd with multiple client per worker"
nohup ssh ubuntu@vm-8-1 "rm ~/output/* ; dstat --output ~/output/sync_stats_1.csv -cdn" &
nohup ssh ubuntu@vm-8-2 "rm ~/output/* ; dstat --output ~/output/sync_stats_2.csv -cdn" &
nohup ssh ubuntu@vm-8-3 "rm ~/output/* ; dstat --output ~/output/sync_stats_3.csv -cdn" &
nohup ssh ubuntu@vm-8-4 "rm ~/output/* ; dstat --output ~/output/sync_stats_4.csv -cdn" &
nohup ssh ubuntu@vm-8-5 "rm ~/output/* ; dstat --output ~/output/sync_stats_5.csv -cdn" &

# testdistributed.py is a client that can run jobs on the cluster.
# please read testdistributed.py to understand the steps defining a Graph and
# launch a session to run the Graph
python synchronoussgd.py

# defined in tfdefs.sh to terminate the cluster
terminate_cluster
killall ssh
