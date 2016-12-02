import tensorflow as tf
import time
## TODO enhancement tf.sparse_tensor_dense_matmul(...)

# Number of features
num_features = 33762578
BATCH_SIZE = 1000

g = tf.Graph()
data_dir = "./data/criteo-tfr-big"
file_names = [data_dir + '/tfrecords00']
# file_names = [data_dir + '/tfrecords00', data_dir + '/tfrecords01', data_dir + '/tfrecords02']
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
    label, index, value = get_next_batch(file_names)
    dense_index = tf.sparse_tensor_to_dense(index)
    dense_value = tf.sparse_tensor_to_dense(value)
    dense_index_shape = tf.shape(dense_index)
    unique_x = tf.unique(index.values).y
    unique_x_filtering = tf.reshape(unique_x, shape=[tf.shape(unique_x)[0],1])
    # dense_index = tf.reshape(dense_index, shape=[2, BATCH_SIZE*])
    w = tf.Variable(tf.ones([num_features,1]), name="model")
    # w_filtered = tf.gather_nd(w, unique_x_filtering)
    # w_filtered_sparse = tf.sparse_reorder(tf.SparseTensor(unique_x_filtering, w_filtered, shape=[num_features]))
    # slice_index = []
    # x_filtered = tf.reshape(tf.convert_to_tensor(value.values, dtype=tf.float32), [tf.shape(value)[0], 1])
    total_gradient = tf.SparseTensor([[0]], [0.0], shape=[num_features])
    for i in range(0, BATCH_SIZE):
        slice_index = tf.slice(dense_index, [i,0], [1,-1])
        slice_value = tf.convert_to_tensor(tf.slice(dense_value, [i,0], [1,-1]), dtype=tf.float32)
        l_filtered = tf.convert_to_tensor(tf.slice(label, [i,0], [1,-1]), dtype=tf.int64)
        slice_index = tf.reshape(slice_index, [tf.shape(value)[1], 1])
        w_filtered = tf.reshape(tf.gather(w, slice_index), [tf.shape(value)[1], 1])
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
    total_gradient_index_shape = tf.shape(total_gradient.indices)[0]
    total_gradient_index = tf.reshape(total_gradient.indices, shape=[total_gradient_index_shape])
    total_gradient_value = tf.reshape(tf.mul(total_gradient.values, -0.01), shape=[total_gradient_index_shape, 1])
    update_model = tf.scatter_add(w, total_gradient_index, total_gradient_value)

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
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()

    # Start the queue readers
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Run n iterations
    n = 1000
    ep = 100
    e = 1000
    count = 0
    try:
        start_total = time.time()
        start = time.time()
        for i in range(0, n):
            sess.run(update_model)
            if i%ep == 0:
                print "Time taken for last " + str(ep) + " iteration and batch size " + str(BATCH_SIZE) + " :: " + str(time.time() - start)
                start = time.time()
                count = 0
                for j in range(0,e):
                    output_sign = sess.run(sign_values)
                    if output_sign[0] != output_sign[1]:
                        count+=1
                print "*********Mistakes: " + str(count), str(e) + "**********"
                print "Time in calculating mistakes on test set: " + str(time.time() - start)
                start = time.time()
        print "Time taken for " + str(n) + " iteration and batch size " + str(BATCH_SIZE) + " :: " + str(time.time() - start_total)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:  # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

