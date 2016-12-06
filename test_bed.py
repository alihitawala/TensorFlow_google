import tensorflow as tf
import os
 ## TODO enhancement tf.sparse_tensor_dense_matmul(...)

# Number of features
num_features = 33762578
BATCH_SIZE = 100

g = tf.Graph()


with g.as_default():
    x = tf.constant(5.0, shape=[5, 1])
    w = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0])
    ww = tf.Variable(tf.ones([5,1]), name="x1")
    ttw = tf.Variable(tf.ones([1], dtype=tf.int64), name="counter")
    c = ttw.assign_add([1])
    d = ttw.assign_add([1])
    xw = [tf.mul(x, w), c, d]
    indices = [[1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
          [['a1', 'b1'], ['c1', 'd1']]]
    g = tf.gather(params, indices)
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(ww, zero)
    indices = tf.where(where)
    ind2 = tf.IndexedSlices(tf.constant([3.0]), tf.constant([0], dtype=tf.int64, shape=[1]))
    tf1 = tf.constant([[1,2,3,7,8,9],[7,8,9,10,11,12]])
    tf_merge = tf.unique(tf.concat(0, [tf1[0], tf1[1]]))

    dx = [[1,2,3,4],[5,6,7,8]]
    slice_index = tf.slice(dx, [1,0], [1,-1])
    # assign_op2 = ww.scatter_sub(ind2)
    # max_in_rows = tf.reduce_max(xw, 1)
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        result = sess.run(ind2)
        print result
        result = sess.run(xw)
        print ttw.eval()
        print ttw.eval()
    # print sess.run(max_in_rows)
