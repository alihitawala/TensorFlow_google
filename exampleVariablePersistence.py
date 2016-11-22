import tensorflow as tf
import os

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

# creating a variable on task 0. This is a process running on node vm-48-1
with tf.device("/job:worker/task:0"):
    X = tf.Variable(tf.constant(0.0), name="globalvar")
    pass

# creating variable on the task (process) specified in the command line.
with tf.device("/job:worker/task:%d" % FLAGS.task_index):
    Y = tf.Variable(tf.constant(0.0), name="localvar")
    pass

b = tf.constant(1.0) # this creates an operator that emits a constant tensor
# incrementing the variables. Note that assign_add is a specific function
# implemented only for variables. Variables also inherit all the functions of a
# tensor. Thus, we can use variables are input to operators.
assign1 = X.assign_add(b)
assign2 = Y.assign_add(b)


config = tf.ConfigProto(log_device_placement=True)
with tf.Session("grpc://vm-48-%d:2222" % (FLAGS.task_index + 1), config=config)  as sess:

    tf.train.SummaryWriter("%s/asyncsgd" % (os.environ.get("TF_LOG_DIR")), sess.graph)

    # variables need to be initialized, if not. if you re-initialize the
    # variables, all previously stored data from other sessions will be lost
    if False ==  tf.is_variable_initialized(X) or False == tf.is_variable_initialized(Y):
        sess.run(tf.initialize_all_variables())

    # feel free to increment the loop count, if you want to observe variables for
    # longer durations.
    for i in range(0, 100):
        # session can be run to compute the values of multiple
        # tensors/variables. The variables of interest are provided as a list
        # when invoking run.
        sess.run([assign1, assign2])
        # observe the values in a variable
        print X.eval(), Y.eval()

    sess.close()

