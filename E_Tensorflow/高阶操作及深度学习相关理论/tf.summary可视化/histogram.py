import tensorflow as tf

w = tf.summary.create_file_writer('test_histogram')
with w.as_default():
    for step in range(100):
        # Generate fake "activations".
        activations = [
            tf.random.normal([1000], mean=step, stddev=1),
            tf.random.normal([1000], mean=step, stddev=10),
            tf.random.normal([1000], mean=step, stddev=100),
        ]

        # Write a histogram summary.
        tf.summary.histogram(name="layer1/activate", data=activations[0], step=step)
        tf.summary.histogram("layer2/activate", activations[1], step=step)
        tf.summary.histogram("layer3/activate", activations[2], step=step)
