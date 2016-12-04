import tensorflow as tf
import time

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS
# Number of features
num_features = 33762578

g = tf.Graph()

# Get the list of all files in the input data directory
data_dir = "./data/criteo-tfr-big"
file_names = {
     '0': [data_dir + '/tfrecords00', data_dir + '/tfrecords01', data_dir + '/tfrecords02', data_dir + '/tfrecords03',
           data_dir + '/tfrecords04'],
     '1': [data_dir + '/tfrecords05', data_dir + '/tfrecords06', data_dir + '/tfrecords07', data_dir + '/tfrecords08',
           data_dir + '/tfrecords09'],
     '2': [data_dir + '/tfrecords10', data_dir + '/tfrecords11', data_dir + '/tfrecords12', data_dir + '/tfrecords13',
           data_dir + '/tfrecords14'],
     '3': [data_dir + '/tfrecords15', data_dir + '/tfrecords16', data_dir + '/tfrecords17', data_dir + '/tfrecords18',
           data_dir + '/tfrecords19'], '4': [data_dir + '/tfrecords20', data_dir + '/tfrecords21']}

file_names = {
    '0': [data_dir + '/tfrecords00'],
    '1': [data_dir + '/tfrecords01'],
    '2': [data_dir + '/tfrecords02'],
    '3': [data_dir + '/tfrecords03'],
    '4': [data_dir + '/tfrecords04'],
    '5': [data_dir + '/tfrecords05'],
    '6': [data_dir + '/tfrecords06'],
    '7': [data_dir + '/tfrecords07'],
    '8': [data_dir + '/tfrecords08'],
    '9': [data_dir + '/tfrecords09'],
    '10': [data_dir + '/tfrecords10'],
    '11': [data_dir + '/tfrecords11'],
    '12': [data_dir + '/tfrecords12'],
    '13': [data_dir + '/tfrecords13'],
    '14': [data_dir + '/tfrecords14'],
    '15': [data_dir + '/tfrecords15'],
    '16': [data_dir + '/tfrecords16'],
    '17': [data_dir + '/tfrecords17'],
    '18': [data_dir + '/tfrecords18'],
    '19': [data_dir + '/tfrecords19'],
    '20': [data_dir + '/tfrecords20'],
    '21': [data_dir + '/tfrecords21'],
}

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
    index_array = []
    # Create a model
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.random_uniform([num_features, 1]), name="model")

    # Compute the gradient
    # dense_x = {}
    with tf.device("/job:worker/task:%d" % (FLAGS.task_index)):
        label, index, value = get_next_row(file_names[str(FLAGS.task_index)])
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
        local_gradient = tf.reshape(tf.mul(tf.reshape(local_gradient, tf.shape(value)), -0.01), [tf.shape(value)[0], 1])
        with tf.device("/job:worker/task:0"):
            assign_op = tf.scatter_add(w, index.values, local_gradient)

    # we create an operator to aggregate the local gradients
    # with tf.device("/job:worker/task:0"):
    #     assign_op = tf.scatter_add(w, index.values, local_gradient)
        # dense_gradient = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
        #     [num_features],
        #     local_gradient)
        # dense_gradient = tf.reshape(dense_gradient, [num_features, 1])
        # assign_op = w.assign_add(tf.mul(dense_gradient, -0.01))
        # test_label, test_index, test_value = get_next_row(test_file_names)
        # test_dense_x = get_dense_x(test_index, test_value)

        # test_w_filtered = tf.gather(w, test_index.values)
        # test_x_filtered = tf.convert_to_tensor(test_value.values, dtype=tf.float32)
        # test_x_filtered = tf.reshape(test_x_filtered, [tf.shape(test_value)[0], 1])
        # sign_actual = tf.cast(tf.sign(tf.matmul(tf.transpose(test_w_filtered), test_x_filtered)[0][0]), tf.int64)
        # sign_expected = tf.sign(test_label[0])
        # sign_values = [sign_actual, sign_expected]

    # Create a session
    with tf.Session("grpc://vm-8-%d:2222" % (1)) as sess:
        print "====================*************"
        # only one client initializes the variable
        if FLAGS.task_index == 0:
            sess.run(tf.initialize_all_variables())
        # Start the queue readers
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        # Start the queue readers
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # Run n iterations
        n = 20
        e = 200
        count = 0
        try:
            start_total = time.time()
            print "==============="
            # while not coord.should_stop():
            # Run training steps or whatever
            for i in range(0, n):
                print "-------------"
                print time.time()
                print "-------------"
                start = time.time()
                sess.run(assign_op)
            #     print "Time taken for training iteration " + str(i) + " at : vm-" + str(FLAGS.task_index+1) + " : " + str(time.time() - start)
            #     if FLAGS.task_index == 0 and i % 10 == 0:
            #         start = time.time()
            #         count = 0
            #         for j in range(0,e):
            #             output_sign = sess.run(sign_values)
            #             if output_sign[0] != output_sign[1]:
            #                 count+=1
            #         print "*********Mistakes: " + str(count), str(e) + "**********"
            #         print "Time in calculating mistakes on test set: " + str(time.time() - start)
            print "Total time taken for " + str(n) + " iterations : " + " at : vm-" + str(int(FLAGS.task_index)/5+1) + " : " + str(time.time() - start_total)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:  # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        #sess.close()
