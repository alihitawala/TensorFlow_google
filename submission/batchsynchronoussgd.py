import tensorflow as tf
import time

# Number of features
num_features = 33762578
BATCH_SIZE = 100

g = tf.Graph()
data_dir = "./data/criteo-tfr-big"
file_names = {
    '0': [data_dir + '/tfrecords00', data_dir + '/tfrecords01', data_dir + '/tfrecords02', data_dir + '/tfrecords03',
          data_dir + '/tfrecords04'],
    '1': [data_dir + '/tfrecords05', data_dir + '/tfrecords06', data_dir + '/tfrecords07', data_dir + '/tfrecords08',
          data_dir + '/tfrecords09'],
    '2': [data_dir + '/tfrecords10', data_dir + '/tfrecords11', data_dir + '/tfrecords12', data_dir + '/tfrecords13',
          data_dir + '/tfrecords14'],
    '3': [data_dir + '/tfrecords15', data_dir + '/tfrecords16', data_dir + '/tfrecords17', data_dir + '/tfrecords18',
          data_dir + '/tfrecords19'],
    '4': [data_dir + '/tfrecords20', data_dir + '/tfrecords21']}
test_file_names = [data_dir + '/tfrecords22']

def get_next_batch(file_names):
    filename_queue = tf.train.string_input_producer(file_names, num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read_up_to(filename_queue, BATCH_SIZE)
    features = tf.parse_example(serialized_example,
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
        w = tf.Variable(tf.random_uniform([num_features,1]), name="model")

    # Compute the gradient
    gradients = []
    # dense_x = {}
    for i in range(0, 5):
        with tf.device("/job:worker/task:%d" % i):
            label, index, value = get_next_batch(file_names[str(i)])
            dense_index = tf.sparse_tensor_to_dense(index)
            dense_value = tf.sparse_tensor_to_dense(value)
            dense_index_shape = tf.shape(dense_index)
            unique_x = tf.unique(index.values).y
            unique_x_filtering = tf.reshape(unique_x, shape=[tf.shape(unique_x)[0],1])
            total_gradient = tf.SparseTensor([[0]], [0.0], shape=[num_features])
            for i in range(0, BATCH_SIZE):
                slice_index = tf.slice(dense_index, [i,0], [1,-1])
                slice_value = tf.convert_to_tensor(tf.slice(dense_value, [i,0], [1,-1]), dtype=tf.float32)
                l_filtered = tf.convert_to_tensor(tf.slice(label, [i,0], [1,-1]), dtype=tf.int64)
                slice_index = tf.reshape(slice_index, [tf.shape(value)[1], 1])

                w_filtered = None
                with tf.device("/job:worker/task:0"):
                    try:
                        w_filtered = tf.reshape(tf.gather(w, slice_index), [tf.shape(value)[1], 1])
                    except:
                        print "Error occurred!"
                        continue

                x_filtered = tf.reshape(slice_value, [tf.shape(value)[1], 1])
                gradient = tf.mul(
                            tf.mul(
                                tf.cast(l_filtered, tf.float32),
                                (tf.sigmoid(
                                    tf.mul(
                                        tf.cast(l_filtered, tf.float32),
                                        (
                                            tf.matmul(
                                                tf.transpose(w_filtered),
                                                x_filtered
                                            )
                                        )
                                    )
                                ) - 1)
                            ), x_filtered)
                gradient = tf.reshape(gradient, shape=[tf.shape(slice_index)[0]])
                sparse_gradient = tf.sparse_reorder(tf.SparseTensor(slice_index, gradient, shape=[num_features]))
                total_gradient = tf.sparse_add(sparse_gradient, total_gradient)
            gradients.append(total_gradient)
    with tf.device("/job:worker/task:0"):
        grand_total_gradients = tf.sparse_add(
                    tf.sparse_add(
                        tf.sparse_add(
                            tf.sparse_add(
                                gradients[0], gradients[1]
                            ), gradients[2]),
                        gradients[3]),
                    gradients[4])
        total_gradient_index_shape = tf.shape(grand_total_gradients.indices)[0]
        total_gradient_index = tf.reshape(grand_total_gradients.indices, shape=[total_gradient_index_shape])
        total_gradient_value = tf.reshape(tf.mul(grand_total_gradients.values, -0.01), shape=[total_gradient_index_shape, 1])
        assign_op = tf.scatter_add(w, total_gradient_index, total_gradient_value)

        # testing
        test_label, test_index, test_value = get_next_row(test_file_names)
        # test_dense_x = get_dense_x(test_index, test_value)
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
        e = 20000 # test examples to read for testing the model
        ep = 1000 # after how many iterations should test be run
        try:
            start_total = time.time()
            print "Batch size for this run is :: " + str(BATCH_SIZE)
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
