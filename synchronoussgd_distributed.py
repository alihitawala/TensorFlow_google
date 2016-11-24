import tensorflow as tf
import time


# Number of features
num_features = 33762578

g = tf.Graph()

# Get the list of all files in the input data directory
data_dir = "./data/criteo-tfr-tiny"
file_names = {
    '0': [data_dir + '/tfrecords00', data_dir + '/tfrecords01', data_dir + '/tfrecords02', data_dir + '/tfrecords03',
          data_dir + '/tfrecords04'],
    '1': [data_dir + '/tfrecords05', data_dir + '/tfrecords06', data_dir + '/tfrecords07', data_dir + '/tfrecords08',
          data_dir + '/tfrecords09'],
    '2': [data_dir + '/tfrecords10', data_dir + '/tfrecords11', data_dir + '/tfrecords12', data_dir + '/tfrecords13',
          data_dir + '/tfrecords14'],
    '3': [data_dir + '/tfrecords15', data_dir + '/tfrecords16', data_dir + '/tfrecords17', data_dir + '/tfrecords18',
          data_dir + '/tfrecords19'], '4': [data_dir + '/tfrecords20', data_dir + '/tfrecords21']}
test_file_names = [data_dir + '/tfrecords22']


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
    dense_x = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
            [num_features],
            tf.sparse_tensor_to_dense(value))
    return tf.reshape(dense_x, [num_features, 1]), label

with g.as_default():
    # Create a model
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.zeros([num_features, 1]), name="model")
        tt = tf.Variable(tf.zeros([num_features, 1]), name="model")

    # Compute the gradient
    gradients = []
    # dense_x = {}
    for i in range(0, 1):
        with tf.device("/job:worker/task:%d" % i):
            # reader = tf.ones([10, 1], name="operator_%d" % i)
            dense_x, label_1 = get_dense_x(file_names[str(i)])
            local_gradient = tf.mul(
                    tf.mul(
                        tf.cast(label_1, tf.float32),
                        (tf.sigmoid(
                            tf.mul(
                                tf.cast(label_1, tf.float32),
                                tf.matmul(
                                    tf.transpose(w),
                                    dense_x
                                )
                            )
                        ) - 1)
                    ), dense_x)
            zero = tf.constant(0, dtype=tf.float32)
            where = tf.not_equal(local_gradient, zero)
            indices = tf.where(where)
            masked = tf.boolean_mask(local_gradient, where)
            sparse_gradient = tf.SparseTensor(indices, tf.transpose(masked), [num_features, 1])
            gradients.append(sparse_gradient)

    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
        dense_gradients = []
        for g in gradients:
            dense_grad = tf.sparse_tensor_to_dense(g)
            dense_gradients.append(dense_grad)
        aggregator = tf.add_n(dense_gradients)
        assign_op = w.assign_add(tf.mul(aggregator, -0.1))
        test_dense_x, test_label = get_dense_x(test_file_names)
        loss = tf.mul(
                -1.0,
                tf.cast(tf.log(
                    tf.sigmoid(
                        tf.mul(
                            tf.cast(test_label, tf.float32),
                            tf.matmul(
                                tf.transpose(w),
                                test_dense_x
                            )
                        )
                    )
                ), tf.float32))
        sign_actual = tf.cast(tf.sign(tf.matmul(tf.transpose(tt), test_dense_x)[0][0]), tf.int64)
        sign_expected = tf.sign(test_label[0])
        sign_values = [sign_actual, sign_expected]

    # Create a session
    with tf.Session("grpc://vm-8-1:2222") as sess:
        sess.run(tf.initialize_all_variables())
        # Start the queue readers
        tf.train.start_queue_runners(sess=sess)
        # Run n iterations
        n = 2000
        e = 20
        count = 0
        for i in range(0, n):
            output = sess.run(assign_op)
            print (output)
            if i % 10 == 0:
                start = time.time()
                count = 0
                for j in range(0,e):
                    output_sign = sess.run(sign_values)
                    if output_sign[0] != output_sign[1]:
                        count+=1
                print "*********" + str(count), str(e) + "**********"
                # loss_out = sess.run(loss)
                print "Time :: " + str(time.time() - start)
        sess.close()