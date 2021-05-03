import tensorflow as tf
import tf_encrypted as tfe

@tfe.local_computation('prediction-client')
def provide_input():
    # normal TensorFlow operations can be run locally
    # as part of defining a private input, in this
    # case on the machine of the input provider
    return tf.ones(shape=(5, 10))

x = provide_input()

model = tfe.keras.Sequential([
    tfe.keras.layers.Dense(512, batch_input_shape=x.shape),
    tfe.keras.layers.Activation('relu'),
    tfe.keras.layers.Dense(10),
])

# get prediction input from client
logits = model(x)

with tfe.Session() as sess:
    result = sess.run(logits.reveal())