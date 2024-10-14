import tensorflow as tf

test_summary_writer = tf.summary.create_file_writer(logdir='test_scala')
with test_summary_writer.as_default():
    # Write a scalar summary.
    tf.summary.scalar(name='loss', data=0.345, step=1)
    tf.summary.scalar('loss', 0.234, step=2)
    tf.summary.scalar('loss', 0.123, step=3)
