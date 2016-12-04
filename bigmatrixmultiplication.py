"""
A solution to finding trace of square of a large matrix using a single device.
We are able to circumvent OOM errors, by generating sub-matrices. TensorFlow
runtime, is able to schedule computation on small sub-matrices without
overflowing the available RAM.
"""

import tensorflow as tf
import os
import sys

class Pair:
    def __init__(self,i,j):
        self.i = i
        self.j = j

    def __hash__(self):
        return hash((self.i, self.j))

    def __eq__(self, other):
        return (self.i, self.j) == (other.i, other.j)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

def getMachineId(i, j, current_task_id):
    pair = Pair(i,j)
    reverse_pair = Pair(j,i)
    if pair in machine_mapping:
        return machine_mapping[pair]
    elif reverse_pair in machine_mapping:
        return machine_mapping[pair]
    else:
        machine_mapping[pair] = current_task_id
        if i == j:
            count[current_task_id] += 1
        else:
            machine_mapping[reverse_pair] = current_task_id
            count[current_task_id] += 2
        return current_task_id

machine_mapping = {}
count = {}
N = 100000 # dimension of the matrix
d = int(sys.argv[1]) # number of splits along one dimension. Thus, we will have 100 blocks
print "value of d is: ", d
M = int(N / d)

BLOCKS_PER_MACHINE = int(d*d/5)

def create_cache():
    current_machine = 0
    for i in range(0,5):
        count[i] = 0
    for i in range(0,d):
        for j in range(0,d):
            if count[current_machine] < BLOCKS_PER_MACHINE or current_machine == 4:
                pass
            else:
                current_machine+=1
            getMachineId(i, j, current_machine)
create_cache()
tf.logging.set_verbosity(tf.logging.DEBUG)
tf.set_random_seed(1024)

def get_block_name(i, j):
    return "sub-matrix-"+str(i)+"-"+str(j)

def get_intermediate_trace_name(i, j):
    return "inter-"+str(i)+"-"+str(j)

# Create a new graph in TensorFlow. A graph contains operators and their
# dependencies. Think of Graph in TensorFlow as a DAG. Graph is however, a more
# expressive structure. It can contain loops, conditional execution etc.
g = tf.Graph()

with g.as_default(): # make our graph the default graph

    # in the following loop, we create operators that generate individual
    # sub-matrices as tensors. Operators and tensors are created using functions
    # like tf.random_uniform, tf.constant are automatically added to the default
    # graph.
    matrices = {}
    for i in range(0,d):
        for j in range(0,d):
            #sys.stdout.write(str(machine_mapping[Pair(i,j)]) + ',' + str(i) +',' + str(j) + '   ')
            with tf.device("/job:worker/task:%d" % machine_mapping[Pair(i,j)]):
                matrix_name = get_block_name(i, j)
                matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)
       # print ""
    # In order the
    # In this loop, we create 100 "matmul" operators that does matrix
    # multiplication. Each "matmul" operator, takes as input two tensors as input.
    # we also create 100 "trace" operators, that takes the output of "matmul" an
    # computes the trace of the martix. Tensorflow defines a trace function;
    # however, when you observe the graph using "tensorboard" you will see that the
    # trace operator is actually implements as multiple small operators.
    print "Matrix block allocation completed."
    intermediate_traces = {}
    for i in range(0, d):
        for j in range(0, d):
           # sys.stdout.write(str(machine_mapping[Pair(i,j)]) + ',' + str(i) +',' + str(j) + '   ')
            with tf.device("/job:worker/task:%d" % machine_mapping[Pair(i,j)]):
                A = matrices[get_block_name(i, j)]
                B = matrices[get_block_name(j, i)]
                intermediate_traces[get_intermediate_trace_name(i, j)] = tf.trace(tf.matmul(A, B))
        #print ""

    print "Intermediate block allocation completed"
    # here, we add a "add_n" operator that takes output of the "trace" operators as
    # input and produces the "retval" output tensor.
    with tf.device("/job:worker/task:0"):
        retval = tf.add_n(intermediate_traces.values())

    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://vm-8-2:2222", config=config) as sess:
        result = sess.run(retval)
        tf.train.SummaryWriter("%s/bigmatrixmultiplication" % (os.environ.get("TF_LOG_DIR")), sess.graph)
        sess.close()
        print "Success"
        print result
