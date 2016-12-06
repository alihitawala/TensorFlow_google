import tensorflow as tf
import os
 ## TODO enhancement tf.sparse_tensor_dense_matmul(...)

# Number of features
num_features = 33762578

g = tf.Graph()
data_dir = "./data/criteo-tfr-tiny"
file_names = [data_dir + '/tfrecords00']


def get_dense_x(file_names):
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
    # dense_x = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
    #         [num_features],
    #         tf.sparse_tensor_to_dense(value))
    # return tf.reshape(dense_x, [num_features, 1]), label
    return value, label

with g.as_default():
    dense_x, label = get_dense_x(file_names)
    # Create a model
    w = tf.Variable(tf.zeros([num_features, 1]), name="model")

    # Compute the gradient
    gradient = tf.Variable(tf.ones([num_features, 1]), dtype=tf.float32)
    # gradient = tf.mul(tf.mul(label, tf.sigmoid(tf.mul(label, tf.matmul(tf.transpose(w), dense_feature) - 1))), value)
    left = tf.mul(
                tf.cast(label, tf.float32),
                (tf.sigmoid(
                    tf.mul(
                        tf.cast(label, tf.float32),
                        (
                            tf.sparse_reduce_sum(tf.SparseTensor.__mul__(dense_x,w))
                            # tf.sparse_tensor_dense_matmul(
                            #     dense_x,
                            #     w
                            # )
                        )
                    )
                ) - 1)
            )
    left = tf.reshape(left, [1,1])
    gradient = tf.sparse_tensor_dense_matmul(dense_x, left)
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(gradient, zero)
    indices = tf.where(where)
    masked = tf.boolean_mask(gradient, where)
    sparse_gradient = tf.SparseTensor(indices, tf.transpose(masked), [num_features, 1])

    # Update the model
    update_model = w.assign_add(tf.mul(gradient, -0.1))

    # Create a session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Start the queue readers
    tf.train.start_queue_runners(sess=sess)

    # Run n iterations
    n = 2000
    count = 0
    for i in range(0, n):
        # print sum(out)
        # output = sess.run(w)
        output_x = sess.run(dense_x)
        # output_g = sess.run(sparse_gradient)
        print sum(output_x)
