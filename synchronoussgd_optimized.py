import tensorflow as tf
import time
 ## TODO enhancement tf.sparse_tensor_dense_matmul(...)

# Number of features
num_features = 33762578

g = tf.Graph()
data_dir = "./data/criteo-tfr-big"
file_names = [data_dir + '/tfrecords00']
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
    label, index, value = get_next_row(file_names)

    # test = tf.gather_nd(params=value, indices=index)
    # Convert to dense feature
    # dense_x = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
    #         [num_features],
    #         tf.sparse_tensor_to_dense(value))
    #
    # dense_x = tf.reshape(dense_x, [num_features, 1])
    # Create a model
    w = tf.Variable(tf.random_uniform([num_features, 1], -1, 1), name="model")
    w_filtered = tf.gather(w, index.values)
    x_filtered = tf.reshape(tf.convert_to_tensor(value.values, dtype=tf.float32), [tf.shape(value)[0], 1])
    l_filtered = label
    # Compute the gradient
    # gradient = tf.Variable(tf.ones([37, 1]), dtype=tf.float32)
    # gradient = tf.mul(tf.mul(label, tf.sigmoid(tf.mul(label, tf.matmul(tf.transpose(w), dense_feature) - 1))), value)
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
    # zero = tf.constant(0, dtype=tf.float32)
    # where = tf.not_equal(gradient, zero)
    # indices = tf.where(where)
    # masked = tf.boolean_mask(gradient, where)
    # sparse_gradient = tf.SparseTensor(indices, tf.transpose(masked), [num_features, 1])
    #
    # # Update the model
    # dense_gradient = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
    #         [num_features],
    #         gradient)
    # dense_gradient = tf.reshape(dense_gradient, [num_features, 1])
    # x = tf.IndexedSlices(gradient, tf.sparse_tensor_to_dense(index) ,dense_shape=tf.constant([num_features, 1]))
    # rx = tf.rank(x)
    update_model = tf.scatter_add(w, index.values, tf.reshape(tf.mul(gradient, -0.01), shape=[tf.shape(value)[0], 1]))


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

    # Start the queue readers
    tf.train.start_queue_runners(sess=sess)

    # Run n iterations
    n = 10000
    e = 2000
    count = 0
    start_total = time.time()
    for i in range(0, n):
        # start_time = time.time()
        output_g = sess.run(update_model)
        # print "Time taken for 1 iteration :: " + str(i) + ' -- ' + str(time.time() - start_time)
        # print (output_g)
        # print (output_g.indices)
        # print (output_g)
        if i % 100 == 0 :
            start = time.time()
            count = 0
            for j in range(0,e):
                output_sign = sess.run(sign_values)
                if output_sign[0] != output_sign[1]:
                    count+=1
            print "*********Mistakes: " + str(count), str(e) + "**********"
            # loss_out = sess.run(loss)
            print "Time in calculating mistakes on test set: " + str(time.time() - start)
    print "Time taken for " + str(n) + " iteration :: " + str(time.time() - start_total)
