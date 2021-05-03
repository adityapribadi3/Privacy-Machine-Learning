#!/usr/bin/env python2

# hack to import tensorspdz from parent directory
# - https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))


from datetime import datetime

from config import MASTER, SESSION_CONFIG, TENSORBOARD_DIR
from tensorspdz import *

from tensorflow.python.client import timeline


# Inputs
input_x, x = define_input((100,100))
input_y, y = define_input((100,100))
input_z, z = define_input((100,100))

# Computation
# v = reveal(dot(x, y) + dot(x, z))
# v = reveal(square(x) + square(y) + square(z))
v = reveal(square(x))

# Actual inputs
X = np.random.randn(100,100)
Y = np.random.randn(100,100)
Z = np.random.randn(100,100)

# Decomposed values outside Tensorflow
inputs = dict(
    [ (xi, Xi) for xi, Xi in zip(input_x, decompose(encode(X))) ] +
    [ (yi, Yi) for yi, Yi in zip(input_y, decompose(encode(Y))) ] +
    [ (zi, Zi) for zi, Zi in zip(input_z, decompose(encode(Z))) ]
)

# Run computation using Tensorflow
with tf.Session(MASTER, config=SESSION_CONFIG) as sess:

    writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    sess.run(tf.global_variables_initializer())
    
    durations = []
    for i in range(10):

        start = datetime.now()
        res = sess.run(
            v,
            inputs,
            options=run_options,
            run_metadata=run_metadata
        )
        end = datetime.now()
        durations.append(end - start)

        writer.add_run_metadata(run_metadata, 'run-{}'.format(i))

        chrome_trace = timeline \
            .Timeline(run_metadata.step_stats) \
            .generate_chrome_trace_format()
        with open('{}/timeline_{}.ctr.json'.format(TENSORBOARD_DIR, i), 'w') as f:
            f.write(chrome_trace)

    for duration in durations:
        print(duration)

    writer.close()

# Recover result outside Tensorflow
V = decode(recombine(res))

# expected = np.dot(X, Y) + np.dot(X, Z)
# expected = X * X + Y * Y + Z * Z
expected = X * X
actual   = V
diff = expected - actual
assert (abs(diff) < 1e-2).all(), abs(diff).max()