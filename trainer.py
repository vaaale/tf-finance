import tensorflow as tf
import numpy as np

lstm_size = 200
batch_size = 100


def train(train_x, train_y):
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # We do 10 iterations (steps) where we grab an example from the CSV file.
        for iteration in range(1, 11):
            # Our graph isn't evaluated until we use run unless we're in an interactive session.
            example, label = sess.run([train_x, train_y])

            print("Iteration ", iteration)
            print(example.shape, label.shape)
        coord.request_stop()
        coord.join(threads)




if __name__ == "__main__":
    data = [x for x in np.random.rand(3)]

    print data