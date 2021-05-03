import tensorflow as tf
import tf_encrypted as tfe

def provide_input() -> tf.Tensor:
    return tf.random_normal(shape=(10,))

inputs = [
            tfe.define_private_input('inputter-0', provide_input),
            tfe.define_private_input('inputter-1', provide_input),
            tfe.define_private_input('inputter-2', provide_input),
            tfe.define_private_input('inputter-3', provide_input),
            tfe.define_private_input('inputter-4', provide_input)
        ]

result = tfe.add_n(inputs) / len(inputs)

def receive_output(average: tf.Tensor) -> tf.Operation:
    return tf.print("Average:", average)

result_op = tfe.define_output('result-receiver', result, receive_output)

with tfe.Session() as sess:
    sess.run(result_op)