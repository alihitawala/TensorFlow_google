"""
    This is the synchronous implementation of the SGD
    Number of clients invoked for this implementation is 20 (4 per worker). This was done to get more parallelism.
    Implementation with 1 clients per worker is also included in the grader directory.
    Iteration time is about 1.4 sec implies = 0.07 sec to get update from one training example.
"""
import tensorflow as tf
import time

# Number of features
num_features = 33762578
learning_rate = -0.01

g = tf.Graph()

# Get the list of all files in the input data directory
data_dir = "./data/criteo-tfr-big"
file_names = {
    '0': [data_dir + '/tfrecords00'],
    '1': [data_dir + '/tfrecords01'],
    '2': [data_dir + '/tfrecords02'],
    '3': [data_dir + '/tfrecords03'],

    '4': [data_dir + '/tfrecords05'],
    '5': [data_dir + '/tfrecords06'],
    '6': [data_dir + '/tfrecords07'],
    '7': [data_dir + '/tfrecords08', data_dir + '/tfrecords09'],

    '8': [data_dir + '/tfrecords10'],
    '9': [data_dir + '/tfrecords11'],
    '10': [data_dir + '/tfrecords12'],
    '11': [data_dir + '/tfrecords13', data_dir + '/tfrecords14'],

    '12': [data_dir + '/tfrecords15'],
    '13': [data_dir + '/tfrecords16'],
    '14': [data_dir + '/tfrecords17'],
    '15': [data_dir + '/tfrecords18', data_dir + '/tfrecords19'],

    '16': [data_dir + '/tfrecords04'],
    '17': [data_dir + '/tfrecords09'],
    '18': [data_dir + '/tfrecords20'],
    '19': [data_dir + '/tfrecords21'],
}
NUM_WORKERS = 20
test_file_names = [data_dir + '/tfrecords22']

def get_next_row(file_names):
    filename_queue = tf.train.string_input_producer(file_names, num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
            features={
                'label' : tf.FixedLenFeature([1], dtype=tf.int64),
                'index' : tf.VarLenFeature(dtype=tf.int64),
                'value' : tf.VarLenFeature(dtype=tf.float32),
                }
            )
    label = features['label']
    index = features['index']
    value = features['value']
    return label, index, value


def get_dense_x(index, value):
    dense_x = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
            [num_features],
            tf.sparse_tensor_to_dense(value))
    return tf.reshape(dense_x, [num_features, 1])

tf.set_random_seed(1024)

with g.as_default():
    # Create a model
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.random_uniform([num_features, 1], -1, 1), name="model")

    # Compute the gradient
    gradients = []
    # dense_x = {}
    for i in range(0, NUM_WORKERS):
        with tf.device("/job:worker/task:%d" % i):
            label, index, value = get_next_row(file_names[str(i)])
            w_filtered = None
            with tf.device("/job:worker/task:0"):
                w_filtered = tf.gather(w, index.values)
            x_filtered = tf.reshape(tf.convert_to_tensor(value.values, dtype=tf.float32), [tf.shape(value)[0], 1])
            l_filtered = label
            local_gradient = tf.mul(
                    tf.mul(
                        tf.cast(l_filtered, tf.float32),
                        (tf.sigmoid(
                            tf.mul(
                                tf.cast(l_filtered, tf.float32),
                                tf.matmul(
                                    tf.transpose(w_filtered),
                                    x_filtered
                                )
                            )
                        ) - 1)
                    ), x_filtered)
            gradients.append(tf.SparseTensor(
                    indices=tf.reshape(index.values, shape=[tf.shape(value)[0], 1]),
                    values=tf.mul(tf.reshape(local_gradient, tf.shape(value)), learning_rate),
                    shape=[num_features]
                )
            )

    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
        # sparse_0 = tf.SparseTensor(indices=gradients[0][1].values, values=gradients[0][0], shape=[num_features, 1])
        total_gradient = tf.sparse_add(gradients[0], gradients[1])
        for i in range(2, NUM_WORKERS) :
            total_gradient = tf.sparse_add(total_gradient, gradients[i])
        total_values = total_gradient.values
        index_total = tf.reshape(total_gradient.indices, shape=tf.shape(total_values))
        gradient_total = tf.reshape(total_values, [tf.shape(total_values)[0],1])
        assign_op = tf.scatter_add(w, index_total, gradient_total)

        test_label, test_index, test_value = get_next_row(test_file_names)
        test_w_filtered = tf.gather(w, test_index.values)
        test_x_filtered = tf.convert_to_tensor(test_value.values, dtype=tf.float32)
        test_x_filtered = tf.reshape(test_x_filtered, [tf.shape(test_value)[0], 1])
        sign_actual = tf.cast(tf.sign(tf.matmul(tf.transpose(test_w_filtered), test_x_filtered)[0][0]), tf.int64)
        sign_expected = tf.sign(test_label[0])
        sign_values = [sign_actual, sign_expected]

    # Create a session
    with tf.Session("grpc://vm-8-1:2222") as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # Start the queue readers
        tf.train.start_queue_runners(sess=sess)
        # Run n iterations
        n = 10 # number of iterations
        e = 100000 # test examples to read for testing the model
        ep = 1000 # after how many iterations should test be run
        count = 0
        try:
            start_total = time.time()
            print "Running " + str(n) + " iterations for sync sgd, with clients = " + str(NUM_WORKERS)
            for i in range(1, n+1):
                start = time.time()
                sess.run(assign_op)
                print "Time taken for training iteration " + str(i) + ": " + str(time.time() - start)
                if i % ep == 0:
                    start = time.time()
                    count = 0
                    for j in range(0,e):
                        output_sign = sess.run(sign_values)
                        if output_sign[0] != output_sign[1]:
                            count+=1
                    print "*********Mistakes: " + str(count), str(e) + "**********"
                    print "Time in calculating mistakes on test set: " + str(time.time() - start)
            print "Total time taken for " + str(n) + " iterations : " + str(time.time() - start_total)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:  # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
