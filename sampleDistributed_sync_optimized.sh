#!/bin/bash

# tfdefs.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source tfdefs.sh

# startserver.py has the specifications for the cluster.
start_cluster startserver.py

echo "Executing the distributed tensorflow synchronous optimized synchronoussgd_distributed_optimized.py"
nohup ssh ubuntu@vm-8-1 "rm ~/output/* ; dstat --output ~/output/sync_stats.csv -cdn" &
nohup ssh ubuntu@vm-8-2 "rm ~/output/* ; dstat --output ~/output/sync_stats.csv -cdn" &
nohup ssh ubuntu@vm-8-3 "rm ~/output/* ; dstat --output ~/output/sync_stats.csv -cdn" &
nohup ssh ubuntu@vm-8-4 "rm ~/output/* ; dstat --output ~/output/sync_stats.csv -cdn" &
nohup ssh ubuntu@vm-8-5 "rm ~/output/* ; dstat --output ~/output/sync_stats.csv -cdn" &

# testdistributed.py is a client that can run jobs on the cluster.
# please read testdistributed.py to understand the steps defining a Graph and
# launch a session to run the Graph
python synchronoussgd_distributed_optimized.py

# defined in tfdefs.sh to terminate the cluster
terminate_cluster
killall ssh
