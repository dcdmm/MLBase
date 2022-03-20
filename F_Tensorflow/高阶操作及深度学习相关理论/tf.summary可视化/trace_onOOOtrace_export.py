import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import os


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


writer = tf.summary.create_file_writer("test_trace_on_trace_export")

# Starts a trace to record computation graphs and profiling information.
'''
graph: If True, enables collection of executed graphs. 
       It includes ones from tf.function invocation and ones from the legacy graph mode. 
       The default is True.
profiler:  If True, enables the advanced profiler. 
           Enabling profiler implicitly enables the graph collection. 
           The profiler may incur a high memory overhead. The default is False.
'''
tf.summary.trace_on(graph=True,  # graphs information
                    profiler=True)  # profile information

model = MyModel()
model.build(input_shape=(None, 28, 28, 1))
img = tf.random.uniform((32, 28, 28, 1))
output = model(img)

with writer.as_default():
    # Stops and exports the active trace as a Summary and/or profile file.
    tf.summary.trace_export(name="MyModel",  # A name for the summary to be written.
                            step=0,
                            # Output directory for profiler. It is required when profiler is enabled when trace was started. Otherwise, it is ignored.
                            profiler_outdir="test_trace_on_trace_export")
