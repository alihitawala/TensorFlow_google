"""
A sample client used to launch jobs on a Distributed TensorFlow cluster.
In this program, we want to find the sum of traces of 5 random matrices.
Each matrix is generated on a different process in the cluster and its traces
are added together.
"""

import tensorflow as tf

# create a empty graph
g = tf.Graph()

# make the graph we created as the default graph. All the variables and
# operators we create will be added to this graph
with g.as_default():

    tf.logging.set_verbosity(tf.logging.DEBUG)

    # this sets the random seed for operations on the current graph
    tf.set_random_seed(1)

    M = 100

    matrices = {} # a container to hold the operators we just created
    for i in range(0, 5):

        with tf.device("/job:worker/task:%d" % i):
            matrix_name = "matrix-"+str(i)
            # create a operator that generates a random_normal tensor of the
            # specified dimensions. By placing this statement here, we ensure
            # that this operator is executed on "/job:worker/task:i". By
            # default, TensorFlow chooses the process to which the client
            # connects to. Feel free to experiment with default or alternate
            # placement strategies.
            matrices[matrix_name] = tf.random_normal([M, M], name=matrix_name)

    # container to hold operators that calculate the traces of individual
    # matrices.
    intermediate_traces = {}
    for i in range(0, 5):

        with tf.device("/job:worker/task:%d" % i):
            A = matrices["matrix-"+str(i)]
            # tf.trace() will create an operator that takes a single matrix
            # as input and calculates it input.
            intermediate_traces["matrix-"+str(i)] = tf.trace(A)

    # sum all the traces
    with tf.device("/job:worker/task:0"):
        retval = tf.add_n(intermediate_traces.values())

    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://vm-48-2:2222", config=config) as sess:
        result = sess.run(retval)
        sess.close()
        print "SUCCESS"
