import tensorflow as tf
import numpy as np
from dataset_reader import load_data


images_batch = tf.placeholder(dtype=tf.float32, shape=[None, 149,])
labels_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])

# simple model
w = tf.get_variable("w1", [149, 22])
y_pred = tf.matmul(images_batch, w)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


# load data entirely into memory
features, labels= load_data()

def data_iterator():
    """ A simple data iterator """
    while True:
        batch_size = 128
        for batch_idx in range(0, len(features), batch_size):
            images_batch = features[batch_idx:batch_idx+batch_size]
            labels_batch = labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch


iter_ = data_iterator()
while True:
    # get a batch of data
    images_batch_val, labels_batch_val = iter_.next()
    # pass it in as through feed_dict
    _, loss_val = sess.run([train_op, loss_mean], feed_dict={
                    images_batch:images_batch_val,
                    labels_batch:labels_batch_val
                    })
    print loss_val


