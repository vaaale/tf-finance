import tensorflow as tf
from dataset_reader import load_data

lstm_size = 200
batch_size = 100


def train(iter):
    batch_x, batch_y = iter.next()
    bull = tf.constant(10, dtype=tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, 149])
    y = tf.placeholder(tf.float32, shape=[None, 22])
    z = tf.add(x, bull)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(100):
            res = sess.run(z, feed_dict={x: batch_x, y: batch_y})
            print res
            print res.shape



def data_iterator(train_x, train_y):
    """ A simple data iterator """
    while True:
        batch_size = 128
        for batch_idx in range(0, len(train_x), batch_size):
            images_batch = train_x[batch_idx:batch_idx+batch_size]
            labels_batch = train_y[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch


if __name__ == "__main__":
    train_x, train_y = load_data()
    iter = data_iterator(train_x, train_y)

    train(iter)