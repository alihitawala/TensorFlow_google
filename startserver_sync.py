"""
A simple script to start tensorflow servers with different roles.
"""
import tensorflow as tf

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
FLAGS = tf.app.flags.FLAGS

#
tf.logging.set_verbosity(tf.logging.DEBUG)

GROUP_NUM=8

clusterSpec = tf.train.ClusterSpec({
    "worker" : [
        "vm-%d-1:2222" % GROUP_NUM,
        "vm-%d-1:2223" % GROUP_NUM,
        "vm-%d-1:2224" % GROUP_NUM,
        "vm-%d-1:2225" % GROUP_NUM,

        "vm-%d-2:2222" % GROUP_NUM,
        "vm-%d-2:2223" % GROUP_NUM,
        "vm-%d-2:2224" % GROUP_NUM,
        "vm-%d-2:2225" % GROUP_NUM,

        "vm-%d-3:2222" % GROUP_NUM,
        "vm-%d-3:2223" % GROUP_NUM,
        "vm-%d-3:2224" % GROUP_NUM,
        "vm-%d-3:2225" % GROUP_NUM,

        "vm-%d-4:2222" % GROUP_NUM,
        "vm-%d-4:2223" % GROUP_NUM,
        "vm-%d-4:2224" % GROUP_NUM,
        "vm-%d-4:2225" % GROUP_NUM,

        "vm-%d-5:2222" % GROUP_NUM,
        "vm-%d-5:2223" % GROUP_NUM
       ]
})

server = tf.train.Server(clusterSpec, job_name="worker", task_index=FLAGS.task_index)
server.join()
