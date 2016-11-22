import tensorflow as tf
import os


# Number of features
num_features = 33762578

g = tf.Graph()
data_dir = "./data/criteo-tfr-tiny"
file_names = [data_dir + '/tfrecords00']
test_file_names = [data_dir + '/tfrecords22']

with g.as_default():
      # Define data queue
    filename_queue = tf.train.string_input_producer(file_names,
            num_epochs=None)
    test_filename_queue = tf.train.string_input_producer(test_file_names,
            num_epochs=None)

    # TFRecordReader for data read
    reader = tf.TFRecordReader()
    test_reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    _, test_serialized_example = test_reader.read(test_filename_queue)

    # Get the features
    features = tf.parse_single_example(serialized_example,
            features={
                'label' : tf.FixedLenFeature([1], dtype=tf.int64),
                'index' : tf.VarLenFeature(dtype=tf.int64),
                'value' : tf.VarLenFeature(dtype=tf.float32),
                }
            )
    # Get the test features
    test_features = tf.parse_single_example(test_serialized_example,
            features={
                'label' : tf.FixedLenFeature([1], dtype=tf.int64),
                'index' : tf.VarLenFeature(dtype=tf.int64),
                'value' : tf.VarLenFeature(dtype=tf.float32),
                }
            )

    label = features['label']
    index = features['index']
    value = features['value']

    test_label = test_features['label']
    test_index = test_features['index']
    test_value = test_features['value']

    #print label
    #print index
    #print value

    # Convert to dense feature
    dense_x = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
            [num_features],
            tf.sparse_tensor_to_dense(value))

    # Convert to dense feature
    test_dense_x = tf.sparse_to_dense(tf.sparse_tensor_to_dense(test_index),
            [num_features],
            tf.sparse_tensor_to_dense(test_value))

    dense_x = tf.reshape(dense_x, [num_features, 1])
    test_dense_x = tf.reshape(test_dense_x, [num_features, 1])
    # Create a model
    w = tf.Variable(tf.ones([num_features, 1]), name="model")

    # Compute the gradient
    gradient = tf.Variable(tf.zeros([num_features, 1]), dtype=tf.float32)
    # gradient = tf.mul(tf.mul(label, tf.sigmoid(tf.mul(label, tf.matmul(tf.transpose(w), dense_feature) - 1))), value)
    gradient = tf.mul(
                    tf.mul(
                        tf.cast(label, tf.float32),
                        tf.sigmoid(
                            tf.mul(
                                tf.cast(label, tf.float32),
                                tf.matmul(
                                    tf.transpose(w),
                                    dense_x
                                ) - 1
                            )
                        )
                    ), dense_x)


    # gradient = tf.reshape(gradient, [num_features, 1])
    # gradient = tf.matmul(tf.transpose(w), dense_feature)
    # print gradient.shape

    # Update the model
    update_model = w.assign_add(tf.mul(gradient, -0.01))
    # update_model = w.assign_add(gradient)

    loss = tf.mul(
                -1.0,
                tf.cast(tf.log(
                    tf.sigmoid(
                        tf.mul(
                            tf.cast(label, tf.float32),
                            tf.matmul(
                                tf.transpose(w),
                                dense_x
                            )
                        )
                    )
                ), tf.float32))
    sign_actual = tf.cast(tf.sign(tf.matmul(tf.transpose(w), test_dense_x)[0][0]), tf.int64)
    sign_expected = tf.sign(test_label[0])
    # Create a session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Start the queue readers
    tf.train.start_queue_runners(sess=sess)

    # Run n iterations
    n = 2000
    count = 0
    for i in range(0, n):
        output = sess.run([update_model, loss])
#        print w.eval(sess)
        if i % 10 == 0:
            for j in range(0,10):
                sign_actual_out, sign_expected_out = sess.run([sign_actual, sign_expected])
                #sign_expected_out = sess.run(sign_expected)
                print sign_expected_out
                count += 1 if sign_actual_out == sign_expected_out else 0
            print "#####",count, i+1
            count = 0
